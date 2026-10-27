"""
Chip-level cocotb testbench for emg_top: the whole design, end to end.

What it does
  1. Trains a 4-gesture INT8 LDA on synthetic EMG with lda_golden_model (same flow as
     the real training script) and loads mu_f, k_f, W, b, cfg_n_ch through the config
     port, then sets core_en.
  2. A model of the daisy-chained ADC pods answers every CONVST: DRDY_N low after 3 us,
     then on CS_N low it shifts out C bytes (channel 0 first, MSB first) on SCLK falling
     edges. The samples are a gesture stream that changes gesture every 2 strides.
  3. The same samples go through LdaCoreRef (the bit-exact golden model). For every
     stride after the first it checks, in order:
        Stage 1+2   the D values written to the x_q RAM        (u_norm.xq_*)
        Stage 3     the 6 class scores                          (u_mac.score_*)
        output      class_valid / class_out at the chip pins
     and at the end: the first decision was masked, primed = 1, overrun = 0.
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


async def load_config(dut, C, mu, k, W, b):
    D = 5 * C
    await cfg_write(dut, 0x7F0, C)
    for f in range(D):
        await cfg_write(dut, f, (int(mu[f]) << 5) | int(k[f]))
    for c in range(G.N_CLASSES):
        for f in range(D):
            await cfg_write(dut, 0x100 + c * D + f, int(W[c, f]) & 0xFF)
        await cfg_write(dut, 0x400 + c, int(b[c]))
    await cfg_write(dut, 0x7F1, 1)


async def adc_model(dut, stream, sent):
    dut.adc_drdy_n.value = 1
    dut.adc_miso.value = 0
    for frame in stream:
        await RisingEdge(dut.adc_convst)
        await Timer(3, unit="us")
        dut.adc_drdy_n.value = 0
        await FallingEdge(dut.adc_cs_n)
        dut.adc_drdy_n.value = 1
        bits = [(int(v) & 0xFF) >> (7 - i) & 1 for v in frame for i in range(8)]
        dut.adc_miso.value = bits[0]
        for bit in bits[1:]:
            await FallingEdge(dut.adc_sclk)
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

    sent = []
    log = dict(start=[], xq=[], scores=[], mac_done=[], cls=[])
    cocotb.start_soon(adc_model(dut, stream, sent))
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

    lat = [m - s for s, m in zip(log["start"], log["mac_done"])]
    dut._log.info(f"{tag}: {len(sent)} ADC frames, {STRIDES} strides; x_q, scores and class_out match the golden "
                  f"model for strides 1..{STRIDES - 1}; first decision masked; Stage 2+3 = {max(lat)} clk_sys cycles "
                  f"(sample period 1015)")
    dut._log.info(f"{tag}: decisions {[c for _, c in log['cls']]}")


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
