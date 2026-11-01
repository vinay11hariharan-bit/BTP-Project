"""
Chip-level cocotb testbench for emg_top: the whole design, end to end (step 5, 1 Nov 2026).

The chip extracts features; the microcontroller (tb/mcu_model.py) talks to it over SPI.

Test "chip"
  1. Calibrates the 8-bit scaling (mu', k) on synthetic EMG with lda_golden_model, then the
     microcontroller writes every setting over SPI: C, ADC settings, bank size B, bank count N,
     warm-up W, the rest threshold T_REST, the 5C {k, mu'} words and the ADS1299 setup commands. It READS EVERY SETTING BACK
     (command 0x03) and checks it, then writes core_en = 1.
  2. A Python model of the daisy-chained ADS1299 devices answers every CONVST (= START):
     DRDY_N low 504.4 us later, then on CS_N low it shifts out, per device, the 24-bit status
     word 1100... and one 24-bit code per channel (SPI mode 1). The codes are made so that the
     chip's INT8 conversion, clip(code >> SHIFT, -128, 127), gives back the INT8 samples.
     The last channel freezes (same 24-bit code every sample) from the last sample of bank 0, so
     it must be flagged dead once the whole window is frozen, and not before. T_REST is set to
     the middle one of the golden model's rest sums, so some read-outs are rest, some are not,
     and one has rest sum = T_REST exactly (not rest: the test is "below").
  3. Each time mcu_drdy rises, the microcontroller reads the frame (command 0x01). The same
     samples go through LdaCoreRef (the bit-exact golden model). For EVERY read-out it checks:
        Stage 1+2   the 5C values written to the x_q RAM       (u_norm.xq_*, finds the stage)
        frame       every byte: counter, flags (rest, dead channel, valid after the warm-up, no
                    link error / settings error / skip / overrun), dead-channel mask, the 5C
                    features, checksum
        rest sum    0x7FB, read right after each frame, = the golden model's rest sum
  With EMG_FILTER=1 (step 5) the chip test runs with the filter on (0x7FC = 1): the codes are a
  large electrode offset per channel + 50 Hz hum + EMG, the output shift (0x7F3) and mu', k are
  calibrated through the golden model's filter, and the filter's 8-bit output is also checked
  for EVERY sample and channel (u_filt.x8_flat). The first output must be 0 (no jump).
Test "settings"
  B outside 1..100 or N outside 1..4: status bit "settings error" = 1 and the start is refused
  (0x7F1 reads 0); valid values clear it. Reset values 98, 2, 1 (B, N, W) and 0 (T_REST, dead
  rule, rest sum). T_REST keeps 21 bits and the dead rule 1 bit. A wrong command does nothing and
  sends 0s; an unused address reads 0.
Test "link"
  The skip rule and the mixed-frame guard:
    slow read  a frame read so slowly that it is still going when the next read-out starts: that
               read-out is skipped, the slow frame is still complete and correct, and the next
               frame has the "skipped" flag and the counter shows the gap
    mixed read a read started while Stage 2 is writing (ignoring data-ready): the chip sends the
               checksum inverted, so the frame fails the check

Settings come from environment variables, read by lda_golden_model.py:
  EMG_BANK_LEN (default 50), EMG_N_BANKS (3), EMG_WARMUP (2), EMG_CHANNELS (8; 8, 12, 16, 20, 24),
  EMG_REQUANT=old (the old calibration rule, k >= 1), EMG_FILTER=1 (filter on in the chip test).
Run (from the RTL folder):   python tb/test_emg_top.py [rtl_dir] [test]
                             EMG_BANK_LEN=7 EMG_N_BANKS=4 EMG_WARMUP=3 python tb/test_emg_top.py
Waveforms: every run writes waveform/emg_top_<test or all>.vcd (open with gtkwave).
Needs: iverilog, cocotb, numpy, scikit-learn.
"""
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lda_golden_model as G                                       # noqa: E402
from mcu_model import (Mcu, expected_frame, flags_byte, checksum,  # noqa: E402
                       FLAG_CFG, FLAG_SKIP, STATUS_ADDR, REST_SUM_ADDR)

CHANNELS = int(os.environ.get("EMG_CHANNELS", 8))
FILTER = os.environ.get("EMG_FILTER") == "1"     # step 5: the chip test runs with the filter on
STRIDES = G.WARMUP + 3          # read-outs to check: the warm-up ones + 3 valid ones
CLK_SYS_NS, CLK_SPI_NS = 1000, 125

# ---- ADS1299 front end ----
DEV_CH = 8                      # channels per ADS1299 device (config 0x7F2)
SHIFT = 11                      # 24-bit code -> INT8 shift (config 0x7F3), see lda_golden_model.ads_code_to_int8
T_CONV_NS = 504400              # single-shot conversion at the 8 kSPS setting: 1033 tCLK of 488.28 ns
ADC_SETUP = [                   # setup commands, sent through config 0x7F4 before core_en
    [0x11],                                  # SDATAC: leave continuous-read mode so registers can be written
    [0x41, 0x00, 0x91],                      # CONFIG1 = 91h: daisy-chain mode, clock output off, 8 kSPS
    [0x43, 0x00, 0xE0],                      # CONFIG3 = E0h: internal reference buffer on
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


def build_case(C, seed, n_strides=None):
    """Calibrate mu', k on synthetic data; return them and a test stream of INT8 samples."""
    rng = np.random.default_rng(seed)
    gains = rng.uniform(4, 40, size=(4, C))
    n_train = max(6 * G.SUB_WIN, 12 * G.SUB_WIN * G.N_BANKS, 600)        # samples per training recording
    train = [(G.to_adc_int8(gains[g] * rng.standard_normal((n_train, C)), 128.0), g)
             for g in range(4) for _ in range(5)]
    raw, _ = G.dataset_features(train, C)
    mu, k = G.calibrate_norm(raw)
    n = (n_strides or STRIDES) * G.SUB_WIN + 2
    seq = [(i // (2 * G.SUB_WIN) + 1) % 4 for i in range(n)]     # gesture 1 for strides 0-1, 2 for 2-3, 3, 0, ...
    stream = np.stack([G.to_adc_int8(gains[g] * rng.standard_normal((1, C)), 128.0)[0] for g in seq])
    return mu, k, stream


def build_case_filter(C, seed, n_strides=None):
    """Filter on (step 5): synthetic 24-bit codes = a large electrode offset per channel (up to half
    of full scale) + 50 Hz hum + EMG whose size depends on the gesture. The filter's output shift
    and mu', k are calibrated on training recordings through the golden model's filter.
    Returns mu, k, the test codes (T x C) and the output shift."""
    rng = np.random.default_rng(seed)
    gains = rng.uniform(4, 40, size=(4, C)) * 2000.0                       # EMG size, in codes
    offset = rng.integers(-(1 << 22), 1 << 22, size=C)
    phase = rng.uniform(0, 2 * np.pi, size=C)

    def record(g_of_t):
        n = len(g_of_t)
        t = np.arange(n)[:, None]
        v = offset + 3000.0 * np.sin(2 * np.pi * 50 * t / 1000 + phase) + gains[g_of_t] * rng.standard_normal((n, C))
        return np.clip(np.rint(v), -(1 << 23), (1 << 23) - 1).astype(np.int64)

    n_train = max(6 * G.SUB_WIN, 12 * G.SUB_WIN * G.N_BANKS, 600)
    train = [(record(np.full(n_train, g)), g) for g in range(4) for _ in range(5)]
    sh = G.filter_shift_for([r for r, _ in train], skip=300)
    raw, _ = G.dataset_features([(G.filter_int8(r, sh), g) for r, g in train], C)
    mu, k = G.calibrate_norm(raw)
    n = (n_strides or STRIDES) * G.SUB_WIN + 2
    codes = record(np.array([(i // (2 * G.SUB_WIN) + 1) % 4 for i in range(n)]))
    return mu, k, codes, sh


async def load_config(mcu, C, mu, k, dev_ch=DEV_CH, shift=SHIFT, setup=ADC_SETUP, verify=True, start=True,
                      t_rest=None, dead_low8=None, filt_on=None):
    """All settings over SPI, then (verify) every one read back, then core_en = 1.
    t_rest / dead_low8: written to 0x7F8 / 0x7F9 when not None (else they keep their reset value 0)."""
    regs = {0x7F0: C, 0x7F2: dev_ch, 0x7F3: shift, 0x7F5: G.SUB_WIN, 0x7F6: G.N_BANKS, 0x7F7: G.WARMUP}
    if t_rest is not None:
        regs[0x7F8] = t_rest
    if dead_low8 is not None:
        regs[0x7F9] = int(dead_low8)
    if filt_on is not None:
        regs[0x7FC] = int(filt_on)
    for a, v in regs.items():
        await mcu.write(a, v)
    words = [G.pack_norm(mu[f], k[f]) for f in range(5 * C)]
    for f, w in enumerate(words):
        await mcu.write(f, w)
    for cmd in setup:                                        # ADS1299 setup, one byte per write
        for j, byte in enumerate(cmd):                       # (an SPI write takes ~230 us, more than
            await mcu.write(0x7F4, byte | (int(j == len(cmd) - 1) << 8))   # the 10 us the chip needs)
    if verify:
        width = {0x7F0: 5, 0x7F2: 4, 0x7F3: 5, 0x7F5: 7, 0x7F6: 3, 0x7F7: 6, 0x7F8: 21, 0x7F9: 1, 0x7FC: 1}
        for a, v in regs.items():
            got = await mcu.read(a)
            assert got == v & ((1 << width[a]) - 1), f"READ-BACK: setting {a:#x} reads {got}, wrote {v}"
        for f, w in enumerate(words):
            got = await mcu.read(f)
            assert got == w, f"READ-BACK: feature {f} {{k, mu'}} reads {got:#010x}, wrote {w:#010x}"
    if start:
        await mcu.write(0x7F1, 1)
        if verify:
            assert await mcu.read(0x7F1) == 1, "READ-BACK: core_en did not turn on"


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
    """Signed value, or None while it is X/Z."""
    return sig.value.to_signed() if sig.value.is_resolvable else None


async def monitor(dut, log, top=None):
    """Record Stage 2 starts and x_q writes, per clk_sys cycle."""
    top = top or dut
    cyc = 0
    while True:
        await RisingEdge(dut.clk_sys)
        await ReadOnly()
        cyc += 1
        if int(top.u_norm.start.value):
            log["start"].append(cyc); log["xq"].append([])
        if int(top.u_norm.xq_we.value) and log["xq"]:
            log["xq"][-1].append((int(top.u_norm.xq_addr.value), sval(top.u_norm.xq_data)))
        if int(top.u_norm.done.value):
            log["done"].append(cyc)
        if "x8" in log and int(top.u_filt.done.value):
            v = top.u_filt.x8_flat.value
            log["x8"].append(int(v) if v.is_resolvable else None)


async def wait_ready(dut):
    """Until data-ready is high (it may already be: a frame published during a read stays ready)."""
    if str(dut.mcu_drdy.value) != "1":
        await RisingEdge(dut.mcu_drdy)


async def frame_reader(dut, mcu, C, frames, n, half_ns=None, rest_log=None):
    """Read a frame each time data-ready is high (5 us later), until n frames are in.
    rest_log: also read the rest sum (0x7FB) right after each frame."""
    while len(frames) < n:
        await wait_ready(dut)
        await Timer(5, unit="us")
        frames.append(await mcu.frame(C, half_ns))
        if rest_log is not None:
            rest_log.append(await mcu.read(REST_SUM_ADDR))


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk_sys, CLK_SYS_NS, unit="ns").start())
    cocotb.start_soon(Clock(dut.clk_spi, CLK_SPI_NS, unit="ns").start())
    mcu = Mcu(dut)
    dut.arst_n.value = 0
    dut.adc_drdy_n.value, dut.adc_miso.value = 1, 0
    await Timer(5, unit="us")
    dut.arst_n.value = 1
    await Timer(5, unit="us")
    return mcu


def expected_frames(golden, flags_extra=None):
    """Frame bytes the chip must send for each read-out of the golden model (no faults)."""
    out = []
    for r, g in enumerate(golden):
        fl = flags_byte(fault=int(any(g["dead"])), valid=int(g["valid"]), rest=int(g["rest"]))
        if flags_extra:
            fl |= flags_extra.get(r, 0)
        out.append(expected_frame(r, fl, g["dead"], g["q"]))
    return out


def frame_diff(got, want, C):
    """Name the first byte that differs."""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            what = ("counter" if i == 0 else "flags" if i == 1 else "dead mask" if i < 5 else
                    f"feature {i - 5} (ch {(i - 5) // 5}, type {(i - 5) % 5})" if i < 5 * C + 5 else "checksum")
            return f"byte {i} ({what}) = {a:#04x}, expected {b:#04x}"
    return f"length {len(got)}, expected {len(want)}"


async def run_chip(dut, C, seed):
    D = 5 * C
    mcu = await reset(dut)
    if FILTER:
        mu, k, codes, shift = build_case_filter(C, seed)
        stream = codes                                           # (only its shape is used below)
    else:
        mu, k, stream = build_case(C, seed)
        codes = int8_to_codes(stream, SHIFT, np.random.default_rng(seed + 1000))
        shift = None
    i0 = G.SUB_WIN - 1                                           # last channel frozen from here on
    stream[i0:, C - 1], codes[i0:, C - 1] = stream[i0, C - 1], codes[i0, C - 1]
    n_sat = int(np.sum(((codes >> SHIFT) > 127) | ((codes >> SHIFT) < -128)))
    n = STRIDES * G.SUB_WIN
    ref0 = G.LdaCoreRef(C, mu, k, filt_shift=shift)
    sums = [d["rest_sum"] for s, c in zip(stream[:n], codes[:n]) if (d := ref0.step(s, c)) is not None]
    t_rest = sorted(sums)[len(sums) // 2]                        # the middle rest sum: some below, one equal
    ref = G.LdaCoreRef(C, mu, k, t_rest=t_rest, filt_shift=shift)
    golden, gold_x8 = [], []
    for s, c in zip(stream[:n], codes[:n]):
        d = ref.step(s, c)
        gold_x8.append(ref.last_x8)
        if d is not None:
            golden.append(d)
    tag = f"C={C} B={G.SUB_WIN} N={G.N_BANKS} W={G.WARMUP}" + (f" filter on, shift {shift}" if FILTER else "")
    sent, frames, rest_log = [], [], []
    log = dict(start=[], xq=[], done=[])
    if FILTER:
        log["x8"] = []
    cocotb.start_soon(adc_model(dut, stream, codes, sent))       # listening before core_en: the chip
    cocotb.start_soon(monitor(dut, log))                         # takes its first sample at once
    if FILTER:
        await load_config(mcu, C, mu, k, t_rest=t_rest, shift=shift, filt_on=True)
    else:
        await load_config(mcu, C, mu, k, t_rest=t_rest)
    cocotb.start_soon(frame_reader(dut, mcu, C, frames, STRIDES, rest_log=rest_log))

    for _ in range(STRIDES * G.SUB_WIN + 2):
        await RisingEdge(dut.adc_convst)
        if len(frames) >= STRIDES:
            break
    for _ in range(40):              # the last frame read takes up to 4 ms at C = 24 (126 bytes at 250 kHz)
        if len(frames) >= STRIDES:
            break
        await Timer(500, unit="us")
    await Timer(1, unit="ms")
    assert len(sent) >= STRIDES * G.SUB_WIN, f"{tag}: only {len(sent)} ADC frames were read by the chip"
    assert len(log["done"]) >= STRIDES, f"{tag}: Stage 2 finished {len(log['done'])} times, expected {STRIDES}"
    assert len(frames) == STRIDES, f"{tag}: the microcontroller got {len(frames)} frames, expected {STRIDES}"

    want = expected_frames(golden)
    for s in range(STRIDES):
        xq = log["xq"][s]
        g = golden[s]
        assert [a for a, _ in xq] == list(range(D)), f"{tag} read-out {s}: x_q write addresses {[a for a, _ in xq][:10]}..."
        bad = [(f, v, int(g["q"][f])) for f, (_, v) in enumerate(xq) if v != g["q"][f]]
        assert not bad, (f"{tag} read-out {s}: STAGE 1/2 mismatch, {len(bad)} of {D} x_q wrong; first: x_q[{bad[0][0]}] "
                         f"(ch {bad[0][0] // 5}, type {bad[0][0] % 5}) = {bad[0][1]}, expected {bad[0][2]}")
        assert frames[s] == want[s], f"{tag} read-out {s}: FRAME mismatch (x_q correct), {frame_diff(frames[s], want[s], C)}"
    assert sum(g["valid"] for g in golden) == STRIDES - G.WARMUP
    want_sums = [g["rest_sum"] for g in golden]
    assert rest_log == want_sums, f"{tag}: REST SUM (0x7FB) per read-out {rest_log}, expected {want_sums}"
    if FILTER:                                                    # the filter's output, sample by sample
        assert len(log["x8"]) >= n, f"{tag}: the filter ran {len(log['x8'])} times, expected {n}"
        for i in range(n):
            got = [None if log["x8"][i] is None else ((log["x8"][i] >> (8 * c)) & 0xFF) for c in range(C)]
            got = [None if v is None else v - 256 * (v >> 7) for v in got]
            assert got == list(gold_x8[i]), f"{tag} sample {i}: FILTER output {got}, expected {list(gold_x8[i])}"
        assert list(gold_x8[0]) == [0] * C, f"{tag}: first filter output {gold_x8[0]} (offsets {codes[0].tolist()})"
    rest = [g["rest"] for g in golden]
    dead_last = [g["dead"][C - 1] for g in golden]
    assert any(rest) and not all(rest) and t_rest in want_sums, f"{tag}: test setup, rest flags {rest} do not cover both cases"
    assert any(dead_last) and not all(dead_last), f"{tag}: test setup, dead flags of the frozen channel {dead_last}"
    assert mcu.x_bits == 0, f"{tag}: the chip drove MISO = X or Z for {mcu.x_bits} bits of the SPI link"
    st = await mcu.status()          # (link error left out: the ADC model has run out of samples by now)
    want_st = (0x20 if STRIDES > G.WARMUP else 0) | (0x02 if any(golden[-1]["dead"]) else 0)
    assert st & 0x7A == want_st, f"{tag}: status {st:#04x} after the run, expected {want_st:#04x} (valid, channel fault of the last read-out)"

    lat = [d - s for s, d in zip(log["start"], log["done"])]
    dut._log.info(f"{tag}: {len(sent)} ADC frames, {STRIDES} read-outs; all settings read back correctly over SPI; "
                  f"every frame ({5 * C + 6} bytes) matches the golden model byte for byte, warm-up ones flagged "
                  f"not valid; Stage 2 = {max(lat)} clk_sys cycles (sample period 1000)")
    dut._log.info(f"{tag}: T_REST = {t_rest}, rest sums {want_sums} (0x7FB matches), rest flags {[int(r) for r in rest]}; "
                  f"frozen channel {C - 1} dead flags {[int(d) for d in dead_last]}")
    if FILTER:
        x8 = np.array(gold_x8[:n])
        dut._log.info(f"{tag}: the filter's output matches the golden model for all {n} samples x {C} channels "
                      f"(largest |x8| {int(np.abs(x8).max())}, {int(np.sum((x8 == 127) | (x8 == -128)))} clipped); "
                      f"offsets up to {int(np.abs(codes[0]).max())} codes give a first output of 0 (no jump)")
    else:
        dut._log.info(f"{tag}: ADS1299 frames, shift {SHIFT}, {n_sat} codes beyond the INT8 range saturated correctly")


@cocotb.test()
async def chip(dut):
    assert CHANNELS in (8, 12, 16, 20, 24), f"CHANNELS = {CHANNELS}: use 8, 12, 16, 20 or 24"
    await run_chip(dut, CHANNELS, seed=CHANNELS)


@cocotb.test()
async def settings(dut):
    """Range checks over SPI, a wrong command, an unused address."""
    mcu = await reset(dut)
    got = [await mcu.read(a) for a in (0x7F5, 0x7F6, 0x7F7)]
    assert got == [98, 2, 1], f"reset values of B, N, W read {got}, expected the old behaviour [98, 2, 1]"
    got = [await mcu.read(a) for a in range(0x7C0, 0x7D1)]
    want = [c & 0xFFFF for c in G.FILTER_COEFS]
    assert got == want, f"reset values of the filter coefficients (0x7C0-0x7D0) read {got}, expected {want}"
    assert await mcu.read(0x7FC) == 0, "the filter is on after reset"
    await mcu.write(0x7C5, -1234 & 0xFFFFFFFF)
    assert await mcu.read(0x7C5) == (-1234 & 0xFFFF), "filter coefficient 0x7C5 does not keep a written value"
    await mcu.write(0x7FC, 3)
    assert await mcu.read(0x7FC) == 1, "the filter-on setting (0x7FC) does not keep 1 bit"
    got = [await mcu.read(a) for a in (0x7F8, 0x7F9, REST_SUM_ADDR)]
    assert got == [0, 0, 0], f"reset values of T_REST, dead rule, rest sum read {got}, expected [0, 0, 0]"
    for addr, val, want in ((0x7F8, 0xFFFFFFFF, 0x1FFFFF), (0x7F8, 1228800, 1228800), (0x7F9, 3, 1), (0x7F9, 0, 0)):
        await mcu.write(addr, val)
        assert await mcu.read(addr) == want, f"setting {addr:#x}: wrote {val:#x}, expected to read {want:#x}"
    await mcu.write(REST_SUM_ADDR, 5)
    assert await mcu.read(REST_SUM_ADDR) == 0, "the rest sum (0x7FB) is read only, but a write changed it"
    for addr, bad, good in ((0x7F5, 0, 50), (0x7F5, 101, 100), (0x7F6, 0, 3), (0x7F6, 5, 4)):
        await mcu.write(0x7F1, 1)
        assert await mcu.read(0x7F1) == 1
        await mcu.write(addr, bad)
        assert await mcu.status() & FLAG_CFG, f"writing {bad} to {addr:#x} did not set the settings-error flag"
        assert await mcu.read(0x7F1) == 0, f"writing {bad} to {addr:#x} did not stop the core"
        await mcu.write(0x7F1, 1)
        assert await mcu.read(0x7F1) == 0, f"core_en accepted while {addr:#x} = {bad}"
        await mcu.write(addr, good)
        assert not await mcu.status() & FLAG_CFG, f"writing {good} to {addr:#x} did not clear the settings-error flag"
        await mcu.write(0x7F1, 0)
    # a wrong command: chip sends 0s and nothing changes
    before = await mcu.read(0x7F0)
    rx = await mcu.xfer([0x55, 0x07, 0xF0, 0, 0, 0, 3, 0, 0])
    assert rx == [0] * 9, f"wrong command 0x55: chip sent {rx}, expected 0s"
    assert await mcu.read(0x7F0) == before, "wrong command 0x55 changed C"
    # a write cut short by CS rising does nothing
    await mcu.xfer([0x02, 0x07, 0xF0, 0, 0, 0])
    assert await mcu.read(0x7F0) == before, "a write cut short changed C"
    assert await mcu.read(0x123) == 0 and await mcu.read(0x7FF) == 0, "an unused address did not read 0"
    dut._log.info("settings: B = 0 / 101 and N = 0 / 5 set the settings-error flag and stop / refuse the core; "
                  "valid values clear it; reset values 98, 2, 1 and 0, 0, 0 (T_REST, dead rule, "
                  "rest sum), 0 (filter on) and the designed coefficients (0x7C0-0x7D0); T_REST keeps 21 bits, the dead rule 1 bit; the rest sum is read only; a wrong command and a cut-short write do nothing; "
                  "unused addresses read 0")


@cocotb.test()
async def link(dut):
    """Skip rule (slow read) and mixed-frame guard (read during Stage 2)."""
    C = 8
    n_str = G.WARMUP + 8
    mu, k, stream = build_case(C, 77, n_strides=n_str)
    mcu = await reset(dut)
    codes = int8_to_codes(stream, SHIFT, np.random.default_rng(1077))
    sent, log = [], dict(start=[], xq=[], done=[])
    cocotb.start_soon(adc_model(dut, stream, codes, sent))
    cocotb.start_soon(monitor(dut, log))
    await load_config(mcu, C, mu, k, verify=False)
    ref = G.LdaCoreRef(C, mu, k)
    golden = [d for s, c in zip(stream[:n_str * G.SUB_WIN], codes) if (d := ref.step(s, c)) is not None]
    period_ns = G.SUB_WIN * 1e6                                   # one read-out per bank
    tag = f"C={C} B={G.SUB_WIN} N={G.N_BANKS} W={G.WARMUP}"

    # 1. normal read of frame 0
    await RisingEdge(dut.mcu_drdy)
    f0 = await mcu.frame(C)
    # 2. slow read of frame 1: lasts 1.5 read-out periods, so read-out 2 starts while CS is low
    await RisingEdge(dut.mcu_drdy)
    half = 1.5 * period_ns / ((5 * C + 7) * 16)
    slow = await mcu.frame(C, half_ns=half)
    # 3. next frame: must be read-out 3 with the skipped flag
    await RisingEdge(dut.mcu_drdy)
    after = await mcu.frame(C)
    await RisingEdge(dut.mcu_drdy)
    after2 = await mcu.frame(C)
    want = expected_frames(golden, {3: FLAG_SKIP})
    assert f0 == want[0], f"{tag}: frame 0 {frame_diff(f0, want[0], C)}"
    assert slow == want[1], f"{tag}: SLOW READ, frame 1 not intact: {frame_diff(slow, want[1], C)}"
    assert after == want[3], f"{tag}: frame after the slow read: {frame_diff(after, want[3], C)} (expected read-out 3, skipped flag)"
    assert after2 == want[4], f"{tag}: next frame: {frame_diff(after2, want[4], C)}"
    # 4. mixed read: start a read right after Stage 2 starts (data-ready ignored)
    n0 = len(log["start"])
    while len(log["start"]) == n0:
        await RisingEdge(dut.clk_sys)
    mixed = await mcu.frame(C)
    assert str(dut.mcu_drdy.value) == "1", f"{tag}: the frame published during the mixed read is not flagged ready"
    body, cks = mixed[:-1], mixed[-1]
    assert cks == (~checksum(body)) & 0xFF, (f"{tag}: MIXED READ, checksum {cks:#04x}, expected the inverted sum "
                                             f"{(~checksum(body)) & 0xFF:#04x} (plain sum {checksum(body):#04x})")
    await wait_ready(dut)
    clean = await mcu.frame(C)
    r = clean[0]
    assert clean == want[r], f"{tag}: frame after the mixed read: {frame_diff(clean, want[r], C)}"
    dut._log.info(f"{tag}: slow read ({2 * half / 1e3:.0f} us per bit, {len(slow)} bytes over 1.5 read-outs) got frame 1 "
                  f"intact; read-out 2 was skipped and frame 3 carried the skipped flag; a read started during "
                  f"Stage 2 got an inverted checksum; frame {r} after it was clean")


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
