"""
Chip-level cocotb testbench for emg_top: the whole design, end to end.

What it does
  1. Trains a 4-gesture INT8 LDA on synthetic EMG with lda_golden_model (same flow as
     the real training script) and loads mu_f, k_f, W, b, cfg_n_ch, the ADC settings
     (8 channels per device, shift SHIFT) and the ADS1299 setup commands through the
     config port, then sets core_en.
  2. A Python model of the daisy-chained ADS1299 devices answers every CONVST (= START):
     DRDY_N low 504.4 us later, then on CS_N low it shifts out, per device, the 24-bit
     status word 1100... and one 24-bit code per channel, MSB first, changing MISO on SCLK
     rising edges (SPI mode 1). It sends exactly the bits the chip reads for C channels.
     The codes are made so that the chip's INT8 conversion, clip(code >> SHIFT, -128, 127),
     gives back the INT8 samples; their low bits are random, and half of the samples at
     127 / -128 get codes far beyond the range, to test the saturation. (This Python model
     ignores the setup commands; tb/test_emg_chain.py checks them with a Verilog model.)
     The samples are a gesture stream that changes gesture every 2 strides.
  3. The same samples go through LdaCoreRef (the bit-exact golden model). For every
     stride after the first it checks, in order:
        Stage 1+2   the D values written to the x_q RAM        (u_norm.xq_*)
        Stage 3     the 6 class scores                          (u_mac.score_*)
        output      class_valid / class_out at the chip pins
     and at the end: the first decision was masked, primed = 1, overrun = 0,
     chan_fault = 0, link_err = 0 and cfg_err = 0.
  A failure message names the first stage that disagrees.

Run (from the RTL folder):   python tb/test_emg_top.py
Channel count: set CHANNELS near the top of this file (8, 12, 16, 20 or 24).
Run time: about 1.5 min at C = 8 up to about 3 min at C = 24.
Waveforms: every run writes waveform/emg_top_<test or all>.vcd (open with gtkwave).
Needs: iverilog, cocotb, numpy, scikit-learn.
"""
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lda_golden_model as G                       # noqa: E402

# ---- set the number of channels here: 8, 12, 16, 20 or 24 (1, 1.5, 2, 2.5 or 3 pods) ----
CHANNELS = 8

STRIDES = 4                     # decisions to simulate (the first one is masked)
CLK_SYS_NS, CLK_SPI_NS = 1000, 125

# ---- ADS1299 front end ----
DEV_CH = 8                      # channels per ADS1299 device (config 0x7F2)
SHIFT = 11                      # 24-bit code -> INT8 shift (config 0x7F3), see lda_golden_model.ads_code_to_int8
T_CONV_NS = 504400              # single-shot conversion at the 8 kSPS setting: 1033 tCLK of 488.28 ns
ADC_SETUP = [                   # setup commands, sent through config 0x7F4 before core_en
    [0x11],                                  # SDATAC: leave continuous-read mode so registers can be written
    [0x41, 0x00, 0x91],                      # CONFIG1 = 91h: daisy-chain mode, clock output off, 8 kSPS
    [0x43, 0x00, 0xE0],                      # CONFIG3 = E0h: internal reference buffer on (check the CONFIG3 table)
    [0x45, 0x07] + [0x60] * 8,               # CH1SET..CH8SET = 60h: gain 24, normal electrode input
    [0x4F, 0x01, 0xFF, 0xFF],                # LOFF_SENSP, LOFF_SENSN = FFh: lead-off detection on every input
    [0x57, 0x00, 0x0A],                      # CONFIG4 = 0Ah: single-shot mode, lead-off comparators on
    [0x10],                                  # RDATAC: data come out after each DRDY_N without a command
]


def int8_to_codes(x8, shift, rng, p_sat=0.5):
    """24-bit ADS1299 codes that the chip turns back into x8: random low bits, and for a share
    p_sat of the samples at 127 / -128, codes beyond the INT8 range (tests the saturation)."""
    x = np.asarray(x8, dtype=np.int64)
    code = (x << shift) + rng.integers(0, 1 << shift, size=x.shape)
    if (128 << shift) < (1 << 23):                                    # saturation codes exist
        hi = (x == 127) & (rng.random(x.shape) < p_sat)
        lo = (x == -128) & (rng.random(x.shape) < p_sat)
        code[hi] = rng.integers(128 << shift, 1 << 23, size=int(hi.sum()))
        code[lo] = rng.integers(-(1 << 23), -128 << shift, size=int(lo.sum()))
    assert np.array_equal(G.ads_code_to_int8(code, shift), x)
    return code


def build_case(C, seed):
    """Train on synthetic data, return model constants and a test stream of int8 samples."""
    rng = np.random.default_rng(seed)
    gains = rng.uniform(4, 40, size=(4, C))
    train = [(G.to_adc_int8(gains[g] * rng.standard_normal((6 * G.SUB_WIN, C)), 128.0), g)
             for g in range(4) for _ in range(5)]
    raw, y = G.dataset_features(train, C)
    mu, k = G.calibrate_norm(raw)
    _, W, b = G.fit_int8_lda(G.normalize(raw, mu, k), y)
    n = STRIDES * G.SUB_WIN + 2
    seq = [(i // (2 * G.SUB_WIN)) % 4 + 1 for i in range(n)]       # gesture 1 for strides 0-1, gesture 2 for 2-3
    stream = np.stack([G.to_adc_int8(gains[g] * rng.standard_normal((1, C)), 128.0)[0] for g in seq])
    return mu, k, W, b, stream


async def cfg_write(dut, addr, data):
    await FallingEdge(dut.clk_sys)
    dut.cfg_we.value, dut.cfg_addr.value, dut.cfg_wdata.value = 1, addr, int(data) & 0xFFFFFFFF
    await FallingEdge(dut.clk_sys)
    dut.cfg_we.value = 0


async def load_config(dut, C, mu, k, W, b, dev_ch=DEV_CH, shift=SHIFT, setup=ADC_SETUP, gap_us=10):
    D = 5 * C
    await cfg_write(dut, 0x7F0, C)
    await cfg_write(dut, 0x7F2, dev_ch)
    await cfg_write(dut, 0x7F3, shift)
    for f in range(D):
        await cfg_write(dut, f, (int(mu[f]) << 5) | int(k[f]))
    for c in range(G.N_CLASSES):
        for f in range(D):
            await cfg_write(dut, 0x100 + c * D + f, int(W[c, f]) & 0xFF)
        await cfg_write(dut, 0x400 + c, int(b[c]))
    for cmd in setup:                                        # ADS1299 setup, one byte per write
        for j, byte in enumerate(cmd):
            await cfg_write(dut, 0x7F4, byte | (int(j == len(cmd) - 1) << 8))
            await Timer(gap_us, unit="us")                   # the chip needs >= 10 us per byte
    await cfg_write(dut, 0x7F1, 1)


def ads_frame_bits(codes, C, dev_ch=DEV_CH, status=None):
    """The bits the chip clocks in for C channels: per device a 24-bit status word, then its
    channels' 24-bit codes, MSB first; the read stops after channel C-1."""
    bits = []
    for d in range(-(-C // dev_ch)):
        st = 0xC00000 if status is None else status[d]
        words = [st] + [int(codes[c]) & 0xFFFFFF for c in range(d * dev_ch, min(C, (d + 1) * dev_ch))]
        bits += [(w >> (23 - i)) & 1 for w in words for i in range(24)]
    return bits


async def adc_model(dut, stream, codes, sent):
    """Python model of the ADS1299 chain (SPI mode 1: MISO changes on SCLK rising edges)."""
    C = stream.shape[1]
    dut.adc_drdy_n.value = 1
    dut.adc_miso.value = 0
    for frame, fcodes in zip(stream, codes):
        await RisingEdge(dut.adc_convst)
        dut.adc_drdy_n.value = 1
        await Timer(T_CONV_NS, unit="ns")
        dut.adc_drdy_n.value = 0
        await FallingEdge(dut.adc_cs_n)
        for bit in ads_frame_bits(fcodes, C):
            await RisingEdge(dut.adc_sclk)
            dut.adc_miso.value = bit
        await RisingEdge(dut.adc_cs_n)
        sent.append(frame)


def sval(sig):
    """Signed value, or None while it is X/Z (expected on the masked first stride)."""
    return sig.value.to_signed() if sig.value.is_resolvable else None


async def monitor(dut, log):
    """Record x_q writes, scores, chip decisions and Stage 2 start times, per clk_sys cycle."""
    cyc = 0
    while True:
        await RisingEdge(dut.clk_sys)
        await ReadOnly()
        cyc += 1
        if int(dut.u_norm.start.value):
            log["start"].append(cyc); log["xq"].append([])
        if int(dut.u_norm.xq_we.value) and log["xq"]:
            log["xq"][-1].append((int(dut.u_norm.xq_addr.value), sval(dut.u_norm.xq_data)))
        if int(dut.u_mac.score_valid.value):
            if int(dut.u_mac.score_cls.value) == 0:
                log["scores"].append([])
            log["scores"][-1].append((int(dut.u_mac.score_cls.value), sval(dut.u_mac.score)))
        if int(dut.u_mac.class_valid.value):
            log["mac_done"].append(cyc)
        if int(dut.class_valid.value):
            v = dut.class_out.value
            log["cls"].append((cyc, int(v) if v.is_resolvable else None))


async def run_chip(dut, C, seed):
    mu, k, W, b, stream = build_case(C, seed)
    D = 5 * C
    cocotb.start_soon(Clock(dut.clk_sys, CLK_SYS_NS, unit="ns").start())
    cocotb.start_soon(Clock(dut.clk_spi, CLK_SPI_NS, unit="ns").start())
    dut.arst_n.value = 0
    dut.cfg_we.value, dut.cfg_addr.value, dut.cfg_wdata.value = 0, 0, 0
    dut.adc_drdy_n.value, dut.adc_miso.value = 1, 0
    await Timer(5, unit="us")
    dut.arst_n.value = 1
    await Timer(5, unit="us")
    await load_config(dut, C, mu, k, W, b)

    codes = int8_to_codes(stream, SHIFT, np.random.default_rng(seed + 1000))
    n_sat = int(np.sum(((codes >> SHIFT) > 127) | ((codes >> SHIFT) < -128)))
    sent = []
    log = dict(start=[], xq=[], scores=[], mac_done=[], cls=[])
    cocotb.start_soon(adc_model(dut, stream, codes, sent))
    cocotb.start_soon(monitor(dut, log))

    ref = G.LdaCoreRef(C, mu, k, W, b)
    golden = [d for s in stream[:STRIDES * G.SUB_WIN] if (d := ref.step(s)) is not None]
    tag = f"C={C}"

    for _ in range(STRIDES * G.SUB_WIN + 2):
        await RisingEdge(dut.adc_convst)
        if len(log["mac_done"]) >= STRIDES:
            break
    await Timer(1, unit="ms")
    assert len(sent) >= STRIDES * G.SUB_WIN, f"{tag}: only {len(sent)} ADC frames were read by the chip"
    assert len(log["mac_done"]) == STRIDES, (f"{tag}: Stage 3 finished {len(log['mac_done'])} times, expected {STRIDES} "
                                            f"(Stage 2 started {len(log['start'])} times)")

    for s in range(1, STRIDES):
        g = golden[s]
        xq = log["xq"][s]
        assert [a for a, _ in xq] == list(range(D)), f"{tag} stride {s}: x_q write addresses {[a for a, _ in xq][:10]}..."
        bad = [(f, v, int(g["q"][f])) for f, (_, v) in enumerate(xq) if v != g["q"][f]]
        assert not bad, (f"{tag} stride {s}: STAGE 1/2 mismatch, {len(bad)} of {D} x_q wrong; first: x_q[{bad[0][0]}] "
                         f"(ch {bad[0][0] // 5}, type {bad[0][0] % 5}) = {bad[0][1]}, expected {bad[0][2]}")
        sc = log["scores"][s]
        assert [c for c, _ in sc] == list(range(G.N_CLASSES)), f"{tag} stride {s}: score classes {[c for c, _ in sc]}"
        bad = [(c, v, g["scores"][c]) for c, v in sc if v != g["scores"][c]]
        assert not bad, f"{tag} stride {s}: STAGE 3 mismatch (x_q correct), score[{bad[0][0]}] = {bad[0][1]}, expected {bad[0][2]}"

    assert not any(c < log["mac_done"][0] + 3 for c, _ in log["cls"]), \
        f"{tag}: class_valid fired for the first decision after reset, which must be masked (primed)"
    assert len(log["cls"]) == STRIDES - 1, f"{tag}: {len(log['cls'])} class_valid pulses, expected {STRIDES - 1}"
    for s, (cyc, cls) in enumerate(log["cls"], start=1):
        assert cyc > log["mac_done"][0], f"{tag}: class_valid fired for the first (masked) decision"
        assert cls == golden[s]["cls"], f"{tag} stride {s}: OUTPUT mismatch, class_out = {cls}, expected {golden[s]['cls']}"
    assert len({golden[s]["cls"] for s in range(1, STRIDES)}) > 1, f"{tag}: test stream did not change the decision"
    assert int(dut.primed.value) == 1, f"{tag}: primed is not set"
    assert int(dut.overrun.value) == 0, f"{tag}: overrun is set (a frame arrived while Stage 1 was busy)"
    assert int(dut.chan_fault.value) == 0, f"{tag}: chan_fault is set (a channel looked dead or lead-off)"
    assert int(dut.link_err.value) == 0, f"{tag}: link_err is set (bad status header or a missing frame)"
    assert int(dut.cfg_err.value) == 0, f"{tag}: cfg_err is set (channel count outside 1..24)"

    lat = [m - s for s, m in zip(log["start"], log["mac_done"])]
    dut._log.info(f"{tag}: {len(sent)} ADC frames, {STRIDES} strides; x_q, scores and class_out match the golden "
                  f"model for strides 1..{STRIDES - 1}; first decision masked; Stage 2+3 = {max(lat)} clk_sys cycles "
                  f"(sample period 1015)")
    dut._log.info(f"{tag}: decisions {[c for _, c in log['cls']]}; ADS1299 frames, shift {SHIFT}, "
                  f"{n_sat} codes beyond the INT8 range saturated correctly")


@cocotb.test()
async def chip(dut):
    assert CHANNELS in (8, 12, 16, 20, 24), f"CHANNELS = {CHANNELS}: use 8, 12, 16, 20 or 24"
    await run_chip(dut, CHANNELS, seed=CHANNELS)


if __name__ == "__main__":
    from cocotb_tools.runner import Icarus

    class IcarusVcd(Icarus):
        """The stock runner passes -none to vvp when waves=False, which suppresses $dumpfile; ask for VCD instead."""
        def _test_command(self):
            return [["-vcd" if a == "-none" else a for a in cmd] for cmd in super()._test_command()]

    rtl_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    sources = sorted(rtl_dir.glob("*.v"))
    print("RTL:", ", ".join(p.name for p in sources))
    runner = IcarusVcd()
    build = Path(__file__).resolve().parent / "sim_build"
    testcase = sys.argv[2] if len(sys.argv) > 2 else None

    # VCD dump: a second top-level module that records every signal under emg_top into RTL/waveform/
    wave_dir = Path(__file__).resolve().parents[1] / "waveform"
    wave_dir.mkdir(exist_ok=True)
    vcd = wave_dir / f"emg_top_{testcase or 'all'}.vcd"
    build.mkdir(exist_ok=True)
    dump_v = build / "wave_dump.v"
    dump_v.write_text(f'module wave_dump();\n  initial begin\n    $dumpfile("{vcd.as_posix()}");\n'
                      f'    $dumpvars(0, emg_top);\n  end\nendmodule\n')
    print("VCD:", vcd)

    runner.build(sources=sources + [dump_v], hdl_toplevel="emg_top", build_dir=build, always=True,
                 timescale=("1ns", "1ps"), build_args=["-s", "wave_dump"])
    runner.test(hdl_toplevel="emg_top", test_module="test_emg_top", test_dir=Path(__file__).parent, build_dir=build,
                testcase=testcase)
