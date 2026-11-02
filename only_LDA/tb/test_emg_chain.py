"""
Chip testbench with a chain of ADS1299 models: emg_top + NPOD copies of tb/ads1299_model.v.

Reuses the training, config loading, ADS1299 setup commands and golden model of
test_emg_top.py. The simulated top level is `tb` (tb/tb_chain.v). The chip itself sends the
setup commands to the devices over MOSI, and the Verilog models decode them, so this test
also checks that every device ends up configured. Before each START, Python only writes each
electrode's 24-bit code and lead-off state to tb.codes / tb.loffp / tb.loffn.

Checks, in order. A failure names the first one that disagrees.
  ADS1299 rules  no datasheet rule broken (START width, CS_N timing, command byte spacing,
                 WREG only after SDATAC) and every device in RDATAC with the setup registers
  SPI timing     the chip never samples MISO while it is x or z
  SPI link       every frame: rx_flat (INT8 per channel), lo_flat (lead-off per channel) and
                 hdr_err match a Python reference of how the chip should parse the bit stream
  Stage 1+2      the D values written to the x_q RAM        (u_top.u_norm.xq_*)
  Stage 3        the 6 class scores                          (u_top.u_mac.score_*)
  dead check     dead_mask flags exactly the channels whose WL window sum is 0
  flags          chan_fault per decision = dead channel or lead-off since the previous decision;
                 link_err per decision = bad status header or lost sample since the previous one
  output         class_valid / class_out, first decision masked, primed = 1, overrun = 0

Faults and edge cases you can inject (the golden model is fed what the chip should receive,
so Stages 1-3 stay bit-exact; the LDA itself was trained on clean data):
  --channels C   larger than the electrodes in the chain: a device is missing. Its data read
                 as 0 and its status word as 0000, so link_err and chan_fault must go to 1
  --cfg-dev-ch N chip told N channels per device. If N differs from --pod-ch the status words
                 land in the wrong place, so link_err must go to 1. 0 or above 8 must act as 8
  --shift S      above 16 must act as 16
  --stuck LIST   electrodes stuck at full scale, -128 for even and +127 for odd channels (like a
                 saturated amplifier); half their codes lie far beyond the INT8 range
  --quiet LIST   electrodes alive but tiny (0, +1 once per stride): must NOT be flagged
  --leadoff LIST electrodes reported off by the ADS1299 (P input for even, N for odd channels).
                 May include electrodes the chip does not read: those must be ignored
  --leadoff-frames A-B   lead-off only in samples A..B (it comes and goes)
  --drop-frame N DRDY_N is held high for sample N, so the chip loses that one sample: link_err
                 must turn on about 1 ms later and turn off again at the next clean decision
  --no-drdy      DRDY_N never falls: link_err must go to 1 within one sample period and no
                 decision may come out
  --late-drdy    conversion takes 1 ms, so the read cannot finish before the next sample:
                 every sample is lost, link_err must go to 1 and no decision may come out
  --cmd-gap US   microseconds between setup command bytes (default 10). Below the chip's
                 10 us bytes are lost, so the devices are not set up: must FAIL
  --bad-c        writes C = 0, 25, 31 and 40: each must raise cfg_err and the start must be
                 refused; a valid C then clears cfg_err and the normal start runs; finally
                 C = 30 written in the middle of a read must stop the core without breaking
                 any ADS1299 rule

Run (from the RTL folder):
  python tb/test_emg_chain.py                                1 device x 8, C = 8
  python tb/test_emg_chain.py --pods 3                       3 x 8, C = 24
  python tb/test_emg_chain.py --pods 4 --pod-ch 4 --channels 14   ADS1299-4, last device partly read
  python tb/test_emg_chain.py --pods 1 --channels 16         device missing
  python tb/test_emg_chain.py --pods 3 --leadoff 4,9 --stuck 16,17 --quiet 2
  python tb/test_emg_chain.py --drop-frame 150
  python tb/test_emg_chain.py --no-drdy
  python tb/test_emg_chain.py --t-d 130                      too slow a DOUT: must FAIL
Options:
  --pods N, --pod-ch N (8, 6 or 4), --channels C (1..24), --cfg-dev-ch N (0..15), --shift S (0..31)
  --stuck, --quiet, --leadoff LIST, --leadoff-frames A-B, --drop-frame N
  --no-drdy, --late-drdy, --cmd-gap US, --bad-c
  --t-conv, --t-h, --t-d, --t-csdod, --t-csdoz (ns, defaults in tb/tb_chain.v)
  --waves         also write waveform/emg_chain_<setup>.vcd (large and slower)
  rtl_dir         folder with the chip's .v files (default: the folder above tb/)
Needs: iverilog, cocotb 2.x, numpy, scikit-learn.
"""
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.simtime import get_sim_time
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly, Timer, ValueChange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lda_golden_model as G                                                    # noqa: E402
from test_emg_top import (STRIDES, CLK_SYS_NS, CLK_SPI_NS, SHIFT, ADC_SETUP,    # noqa: E402
                          build_case, load_config, cfg_write, sval, int8_to_codes)

SETUP_REGS = dict(config1=0x91, config3=0xE0, config4=0x0A, loff_sensp=0xFF, loff_sensn=0xFF)
LATE_T_CONV_NS = 1_000_000.0          # --late-drdy: conversion longer than the 1015 us sample period


def channel_list(text):
    return [int(c) for c in text.split(",") if c.strip()] if text else []


def frame_range(text):
    """'A-B' -> range(A, B+1); '' -> None (all frames)."""
    if not text:
        return None
    a, b = (int(v) for v in text.split("-"))
    return range(a, b + 1)


def inject(stream, stuck, quiet):
    """Electrode values after the stuck / quiet faults (stream is N x C int8)."""
    elec = stream.astype(np.int64).copy()
    for c in stuck:
        elec[:, c] = -128 if c % 2 == 0 else 127       # full scale, both signs (tests saturation too)
    for c in quiet:
        elec[:, c] = 0
        elec[::G.SUB_WIN, c] = 1                       # +1 on the first sample of every stride
    return elec


def status_words(pods, pch, leadoff):
    """24-bit status word of each device: 1100, LOFF_STATP, LOFF_STATN, 0000."""
    out = []
    for d in range(pods):
        lp = sum(1 << i for i in range(pch) if d * pch + i in leadoff and (d * pch + i) % 2 == 0)
        ln = sum(1 << i for i in range(pch) if d * pch + i in leadoff and (d * pch + i) % 2 == 1)
        out.append(0xC00000 | lp << 12 | ln << 4)
    return out


def chip_reference(codes, status, pch, C, dev_ch, shift):
    """How the chip should parse one frame: the devices send status + pch codes each, then the
    chain input gives 0s. The chip reads ceil(C / dev_ch) groups of (status, dev_ch channels),
    stopping after channel C-1. Returns (INT8 per channel, lead-off per channel, header error)."""
    stream = []
    for d, st in enumerate(status):
        stream += [st] + [int(v) & 0xFFFFFF for v in codes[d * pch:(d + 1) * pch]]
    pos, q8, lo, hdr_err = 0, [], [0] * C, False

    def word():
        nonlocal pos
        w = stream[pos] if pos < len(stream) else 0
        pos += 1
        return w

    for g in range(-(-C // dev_ch)):
        st = word()
        hdr_err |= (st >> 20) != 0xC
        for i in range(dev_ch):
            if g * dev_ch + i < C:
                lo[g * dev_ch + i] = (st >> (12 + i) & 1) | (st >> (4 + i) & 1)
        for i in range(dev_ch):
            if g * dev_ch + i >= C:
                break
            w = word()
            q8.append(int(G.ads_code_to_int8(w - (1 << 24) if w >> 23 else w, shift)))
    return q8, lo, hdr_err


async def feed_devices(dut, codes, loff, drop, sent, events):
    """Put each frame's codes and lead-off bits on the electrodes before START (the devices copy
    them at START). Frame `drop` is converted but its DRDY_N is held high, so the chip loses it."""
    i = 0
    while i < len(codes):
        dut.codes.value = sum((int(v) & 0xFFFFFF) << (24 * c) for c, v in enumerate(codes[i]))
        dut.loffp.value, dut.loffn.value = loff[i]
        await RisingEdge(dut.adc_convst)
        if i == drop:
            events["drop_start"] = get_sim_time("ns")
            dut.drdy_block.value = 1
            i += 1
            dut.codes.value = sum((int(v) & 0xFFFFFF) << (24 * c) for c, v in enumerate(codes[i]))
            dut.loffp.value, dut.loffn.value = loff[i]
            await RisingEdge(dut.adc_convst)              # the lost sample's period is over
            dut.drdy_block.value = 0
        await FallingEdge(dut.adc_cs_n)
        await RisingEdge(dut.adc_cs_n)
        sent.append(i)
        i += 1


async def miso_watch(dut, bad):
    """In data reads the chip samples MISO on each SCLK falling edge: record any x or z there."""
    adc = dut.u_top.u_adc
    while True:
        await FallingEdge(dut.adc_sclk)
        if str(adc.en_s.value) == "1" and str(dut.adc_cs_n.value) == "0" and not dut.adc_miso.value.is_resolvable:
            bad.append((get_sim_time("ns"), str(dut.adc_miso.value)))


async def link_monitor(dut, frames):
    """Record rx_flat, lo_flat and hdr_err each time the chip publishes a frame."""
    adc = dut.u_top.u_adc
    while True:
        await ValueChange(adc.frame_tog)
        await ReadOnly()
        frames.append((str(adc.rx_flat.value), str(adc.lo_flat.value), str(adc.hdr_err.value)))


def bytes_of(bits, n):
    """rx_flat bit string (MSB first) -> first n channel bytes as signed ints (None if x/z)."""
    out = []
    for c in range(n):
        b = bits[len(bits) - 8 * (c + 1): len(bits) - 8 * c]
        out.append(int(b, 2) - (256 if b[0] == "1" else 0) if set(b) <= {"0", "1"} else None)
    return out


def flags_of(bits, n):
    """lo_flat bit string (MSB first) -> first n channel flags (None if x/z)."""
    return [int(bits[len(bits) - 1 - c]) if bits[len(bits) - 1 - c] in "01" else None for c in range(n)]


def mask_set(v):
    if not v.is_resolvable:
        return None
    m = int(v)
    return {c for c in range(m.bit_length()) if m >> c & 1}


async def monitor(dut, log):
    """Record x_q writes, scores, dead masks, chip decisions and flags, per clk_sys cycle."""
    top = dut.u_top
    while True:
        await RisingEdge(dut.clk_sys)
        await ReadOnly()
        if int(top.u_norm.start.value):
            log["xq"].append([])
        if int(top.u_norm.xq_we.value) and log["xq"]:
            log["xq"][-1].append((int(top.u_norm.xq_addr.value), sval(top.u_norm.xq_data)))
        if int(top.u_norm.done.value):
            log["dead"].append(mask_set(top.u_norm.dead_mask.value))
        if int(top.u_mac.score_valid.value):
            if int(top.u_mac.score_cls.value) == 0:
                log["scores"].append([])
            log["scores"][-1].append((int(top.u_mac.score_cls.value), sval(top.u_mac.score)))
        if int(top.u_mac.class_valid.value):
            log["mac_done"] += 1
        if str(top.link_err.value) == "1" and log["link_rise"] is None:
            log["link_rise"] = get_sim_time("ns")
        if int(top.class_valid.value):
            v = top.class_out.value
            log["cls"].append(int(v) if v.is_resolvable else None)
            log["fault"].append(str(top.chan_fault.value))
            log["link"].append(str(top.link_err.value))


def check_devices(dut, pods, tag, rules=True):
    """Every device obeyed the datasheet rules and holds the setup registers."""
    viol = [int(dut.pod_viol.value) >> (32 * k) & 0xFFFFFFFF for k in range(pods)]
    if rules:
        assert not any(viol), f"{tag}: ADS1299 RULES broken, violations per device {viol} (see the messages above)"
    for k in range(pods):
        assert int(dut.pod_rdatac.value) >> k & 1, f"{tag}: device {k} is not in RDATAC mode"
        assert int(dut.pod_chset_ok.value) >> k & 1, f"{tag}: device {k} CHnSET registers are not 60h"
        for name, want in SETUP_REGS.items():
            got = int(getattr(dut, "pod_" + name).value) >> (8 * k) & 0xFF
            assert got == want, f"{tag}: device {k} {name.upper()} = {got:02X}h, expected {want:02X}h"
    return viol


async def count_rises(sig, box):
    while True:
        await RisingEdge(sig)
        box[0] += 1


async def bad_c_test(dut, C, mu, k, W, b, pods, tag, codes, loff):
    """C outside 1..24 must raise cfg_err and keep the core from starting; a valid C clears it;
    a bad C written while running stops the core, and the read it cuts short keeps the ADS1299 rules."""
    conv, frames = [0], []
    cocotb.start_soon(count_rises(dut.adc_convst, conv))
    cocotb.start_soon(link_monitor(dut, frames))
    cocotb.start_soon(feed_devices(dut, codes, loff, -1, [], {}))
    core_en = dut.u_top.u_cfg.core_en
    done = []
    for bad in (0, 25, 31, 40):
        await cfg_write(dut, 0x7F0, bad)
        await RisingEdge(dut.clk_sys)
        await ReadOnly()
        assert str(dut.cfg_err.value) == "1", f"{tag}: C = {bad} written but cfg_err = {dut.cfg_err.value}"
        await cfg_write(dut, 0x7F1, 1)                                  # try to start
        await Timer(3, unit="ms")
        assert str(core_en.value) == "0", f"{tag}: C = {bad}, the core started anyway"
        assert conv[0] == 0 and not frames, f"{tag}: C = {bad}, {conv[0]} START pulses / {len(frames)} frames"
        done.append(bad)
    await cfg_write(dut, 0x7F0, C)                                      # fix C
    await RisingEdge(dut.clk_sys)
    await ReadOnly()
    assert str(dut.cfg_err.value) == "0", f"{tag}: valid C = {C} written but cfg_err stays 1"
    await Timer(2, unit="ms")
    assert str(core_en.value) == "0" and conv[0] == 0, f"{tag}: the refused start was remembered and the core started"
    await load_config(dut, C, mu, k, W, b, setup=ADC_SETUP)             # normal start
    check_devices(dut, pods, tag)
    await Timer(5, unit="ms")
    assert str(dut.cfg_err.value) == "0" and conv[0] >= 4 and len(frames) >= 3, \
        f"{tag}: after a valid start: cfg_err {dut.cfg_err.value}, {conv[0]} START pulses, {len(frames)} frames"
    # bad C in the middle of a read: the core must stop, and the cut-short read must keep the rules
    await FallingEdge(dut.adc_cs_n)
    await Timer(20, unit="us")
    t_bad, n_conv = get_sim_time("ns"), conv[0]
    await cfg_write(dut, 0x7F0, 30)
    await Timer(3, unit="ms")
    assert str(dut.cfg_err.value) == "1" and str(core_en.value) == "0", f"{tag}: bad C while running did not stop the core"
    assert conv[0] == n_conv, f"{tag}: {conv[0] - n_conv} START pulses after the core was stopped"
    assert str(dut.adc_cs_n.value) == "1", f"{tag}: CS_N left low after stopping"
    check_devices(dut, pods, tag)                                       # no rule broken by the cut-short read
    dut._log.info(f"{tag}: C = {done} each raised cfg_err and the start was refused (no START, no frame, 3 ms each); "
                  f"C = {C} cleared cfg_err without starting by itself; the normal start then ran; C = 30 written "
                  f"20 us into a read (at {t_bad / 1e3:.0f} us) stopped the core with no ADS1299 rule broken "
                  f"(CS_N held low 3 us before release)")


@cocotb.test()
async def chain(dut):
    env = os.environ
    C, pods, pch = int(env["EMG_CHANNELS"]), int(env["EMG_PODS"]), int(env["EMG_POD_CH"])
    dev_ch, shift = int(env["EMG_CFG_DEV_CH"]), int(env["EMG_SHIFT"])
    stuck, quiet, leadoff = (channel_list(env.get(n, "")) for n in ("EMG_STUCK", "EMG_QUIET", "EMG_LEADOFF"))
    lo_frames = frame_range(env.get("EMG_LO_FRAMES", ""))
    drop = int(env.get("EMG_DROP", "-1"))
    no_drdy, late = env.get("EMG_NO_DRDY") == "1", env.get("EMG_LATE") == "1"
    cmd_gap = float(env.get("EMG_CMD_GAP", "10"))
    n_phys = pods * pch
    D = 5 * C
    tag = f"{pods} device(s) x {pch} electrodes, C={C}"
    missing = list(range(n_phys, C))
    dev_eff = dev_ch if 1 <= dev_ch <= 8 else 8               # what the chip does with 0 / >8
    sh_eff = min(shift, 16)                                     # and with a shift above 16

    mu, k, W, b, stream = build_case(C, seed=C)                  # LDA trained on clean data
    elec = inject(stream, stuck, quiet)                          # INT8 value of each electrode
    rng = np.random.default_rng(2000 + C)
    phys = np.zeros((len(elec), n_phys), dtype=np.int64)         # electrodes that exist
    m = min(C, n_phys)
    phys[:, :m] = elec[:, :m]
    phys[:, m:] = rng.integers(-128, 128, size=(len(elec), n_phys - m))   # unread electrodes
    codes = int8_to_codes(phys, sh_eff, rng)
    lo_on = [lo_frames is None or i in lo_frames for i in range(len(codes))]
    st_on, st_off = status_words(pods, pch, set(leadoff)), status_words(pods, pch, set())
    lp = sum(1 << c for c in leadoff if c % 2 == 0)
    ln = sum(1 << c for c in leadoff if c % 2 == 1)
    loff = [(lp, ln) if on else (0, 0) for on in lo_on]
    ref = [chip_reference(fc, st_on if on else st_off, pch, C, dev_eff, sh_eff) for fc, on in zip(codes, lo_on)]
    faults = bool(stuck or quiet or leadoff or missing or dev_eff != pch or drop >= 0)
    if faults or dev_ch != dev_eff or shift != sh_eff:
        dut._log.info(f"{tag}: missing {missing}, stuck {stuck}, quiet {quiet}, lead-off {leadoff}"
                      f"{' in samples ' + str(lo_frames.start) + '-' + str(lo_frames.stop - 1) if lo_frames else ''}, "
                      f"lost sample {drop if drop >= 0 else 'none'}, chip told {dev_ch} channels per device "
                      f"(acts as {dev_eff}), shift {shift} (acts as {sh_eff})")

    cocotb.start_soon(Clock(dut.clk_sys, CLK_SYS_NS, unit="ns").start())
    cocotb.start_soon(Clock(dut.clk_spi, CLK_SPI_NS, unit="ns").start())
    dut.arst_n.value = 0
    await Timer(5, unit="us")
    dut.arst_n.value = 1
    await Timer(5, unit="us")

    if env.get("EMG_BAD_C") == "1":
        await bad_c_test(dut, C, mu, k, W, b, pods, tag, codes, loff)
        return

    sent, bad_miso, frames, events = [], [], [], {}
    log = dict(xq=[], dead=[], scores=[], mac_done=0, cls=[], fault=[], link=[], link_rise=None)
    cocotb.start_soon(feed_devices(dut, codes, loff, drop, sent, events))
    cocotb.start_soon(miso_watch(dut, bad_miso))
    cocotb.start_soon(link_monitor(dut, frames))
    cocotb.start_soon(monitor(dut, log))
    await load_config(dut, C, mu, k, W, b, dev_ch=dev_ch, shift=shift, setup=ADC_SETUP, gap_us=cmd_gap)
    t_en = get_sim_time("ns")                                    # core_en was just set
    check_devices(dut, pods, tag)

    if no_drdy or late:
        # no complete frame: link_err must rise one sample period after the first START
        while str(dut.link_err.value) != "1" and get_sim_time("ns") - t_en < 5e6:
            await RisingEdge(dut.clk_sys)
        t_err = get_sim_time("ns") - t_en
        await Timer(5, unit="ms")
        assert str(dut.link_err.value) == "1", f"{tag}: NO DATA, link_err never rose"
        assert not frames and not log["cls"], f"{tag}: NO DATA, but {len(frames)} frames / {len(log['cls'])} decisions came"
        check_devices(dut, pods, tag)
        why = "DRDY_N never fell" if no_drdy else "every read was cut off by the next sample (conversion 1 ms)"
        dut._log.info(f"{tag}: {why}: no frame, no decision, link_err rose {t_err / 1e3:.0f} us after core_en "
                      f"and stayed 1; devices configured, no ADS1299 rule broken")
        return

    n = STRIDES * G.SUB_WIN
    seen = [i for i in range(len(codes)) if i != drop]           # samples the chip actually receives
    recv = np.array([ref[i][0] for i in seen], dtype=np.int64)
    core = G.LdaCoreRef(C, mu, k, W, b)
    golden = [d for s in recv[:n] if (d := core.step(s)) is not None]
    raw, _ = G.window_features(recv[:n], C)
    gold_dead = [{c for c in range(C) if raw[s, 5 * c + 1] == 0} for s in range(STRIDES)]

    for _ in range(n + 3):
        await RisingEdge(dut.adc_convst)
        if log["mac_done"] >= STRIDES:
            break
    await Timer(1, unit="ms")

    check_devices(dut, pods, tag)
    # SPI timing
    assert not bad_miso, (f"{tag}: SPI TIMING, the chip sampled MISO = {bad_miso[0][1]} at {bad_miso[0][0]:.0f} ns "
                          f"({len(bad_miso)} bad samples): the device's bit was not valid yet")
    # SPI link
    assert len(sent) >= n, f"{tag}: only {len(sent)} ADC frames were read by the chip"
    assert sent[:len(seen)] == seen[:len(sent)], f"{tag}: the chip read samples {sent[:5]}..., expected {seen[:5]}..."
    assert len(frames) >= len(sent), f"{tag}: {len(sent)} reads but the chip published only {len(frames)} frames"
    for j, i in enumerate(sent):
        q8, lo, hdr = ref[i]
        got_q = bytes_of(frames[j][0], C)
        bad = [(c, g, w) for c, (g, w) in enumerate(zip(got_q, q8)) if g != w]
        assert not bad, (f"{tag} sample {i}: SPI LINK mismatch, {len(bad)} channel(s) wrong; first: channel {bad[0][0]} "
                         f"read as {bad[0][1]}, expected {bad[0][2]}")
        assert set(frames[j][0][:len(frames[j][0]) - 8 * C]) <= {"0"}, f"{tag} sample {i}: rx_flat not 0 above channel {C - 1}"
        got_lo = flags_of(frames[j][1], C)
        assert got_lo == lo, f"{tag} sample {i}: LEAD-OFF flags {got_lo}, expected {lo}"
        assert frames[j][2] == str(int(hdr)), f"{tag} sample {i}: hdr_err = {frames[j][2]}, expected {int(hdr)}"

    # Stages 1-3
    assert log["mac_done"] == STRIDES, f"{tag}: Stage 3 finished {log['mac_done']} times, expected {STRIDES}"
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

    # dead check and flags
    for s in range(1, STRIDES):
        assert log["dead"][s] == gold_dead[s], (f"{tag} stride {s}: DEAD CHECK mismatch, dead_mask flags "
                                                f"{sorted(log['dead'][s]) if log['dead'][s] is not None else 'x'}, "
                                                f"expected {sorted(gold_dead[s])}")
    # chip samples 98s .. 98s+97 arrive between decision s-1 and decision s; a lost sample is
    # noticed just before the chip sample that follows it, so it belongs to that window
    win = [[seen[j] for j in range(G.SUB_WIN * s, G.SUB_WIN * (s + 1))] for s in range(1, STRIDES)]
    lost_in = drop // G.SUB_WIN if drop >= 0 else None
    want_fault = [str(int(bool(gold_dead[s]) or any(any(ref[i][1]) for i in win[s - 1]))) for s in range(1, STRIDES)]
    want_link = [str(int(any(ref[i][2] for i in win[s - 1]) or s == lost_in)) for s in range(1, STRIDES)]
    assert log["fault"] == want_fault, f"{tag}: chan_fault per decision {log['fault']}, expected {want_fault}"
    assert log["link"] == want_link, f"{tag}: link_err per decision {log['link']}, expected {want_link}"
    if drop >= 0:
        delay = (log["link_rise"] - events["drop_start"]) / 1e3
        assert 1000 <= delay <= 1030, f"{tag}: link_err rose {delay:.0f} us after the lost sample's START, expected ~1018 us"

    # output
    assert len(log["cls"]) == STRIDES - 1, f"{tag}: {len(log['cls'])} class_valid pulses, expected {STRIDES - 1}"
    for s, cls in enumerate(log["cls"], start=1):
        assert cls == golden[s]["cls"], f"{tag} stride {s}: OUTPUT mismatch, class_out = {cls}, expected {golden[s]['cls']}"
    if not faults and len({golden[s]["cls"] for s in range(1, STRIDES)}) == 1:
        dut._log.warning(f"{tag}: note, the test stream gave the same gesture every time (test data, not a chip error)")
    assert int(dut.primed.value) == 1, f"{tag}: primed is not set"
    assert int(dut.overrun.value) == 0, f"{tag}: overrun is set (a frame arrived while Stage 1 was busy)"
    assert str(dut.cfg_err.value) == "0", f"{tag}: cfg_err is set"

    n_sat = int(np.sum(((codes[:, :m] >> sh_eff) > 127) | ((codes[:, :m] >> sh_eff) < -128)))
    dut._log.info(f"{tag}: {len(sent)} frames through the ADS1299 chain, devices configured by the chip, no "
                  f"datasheet rule broken, every MISO sample valid, every frame parsed right ({n_sat} saturated codes); "
                  f"x_q, scores, dead_mask, chan_fault, link_err and class_out match for strides 1..{STRIDES - 1}")
    extra = f", link_err rose {delay:.0f} us after the lost sample's START" if drop >= 0 else ""
    dut._log.info(f"{tag}: decisions {log['cls']}, chan_fault {log['fault']}, link_err {log['link']}, "
                  f"dead channels per stride {[sorted(log['dead'][s]) for s in range(1, STRIDES)]}{extra}")
    if faults:
        clean = G.LdaCoreRef(C, mu, k, W, b)
        clean_cls = [d["cls"] for s in stream[:n] if (d := clean.step(s)) is not None][1:]
        dut._log.info(f"{tag}: decisions with every electrode healthy would have been {clean_cls}")


if __name__ == "__main__":
    import argparse
    from cocotb_tools.runner import Icarus, get_results

    p = argparse.ArgumentParser(description="emg_top + ADS1299 chain testbench")
    p.add_argument("rtl_dir", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--pods", type=int, default=1)
    p.add_argument("--pod-ch", type=int, default=8)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--cfg-dev-ch", type=int, default=None)
    p.add_argument("--shift", type=int, default=SHIFT)
    p.add_argument("--stuck", default="")
    p.add_argument("--quiet", default="")
    p.add_argument("--leadoff", default="")
    p.add_argument("--leadoff-frames", default="")
    p.add_argument("--drop-frame", type=int, default=-1)
    p.add_argument("--no-drdy", action="store_true")
    p.add_argument("--late-drdy", action="store_true")
    p.add_argument("--cmd-gap", type=float, default=10.0)
    p.add_argument("--bad-c", action="store_true")
    for t in ("t-conv", "t-h", "t-d", "t-csdod", "t-csdoz"):
        p.add_argument("--" + t, type=float, default=None)
    p.add_argument("--waves", action="store_true")
    a = p.parse_args()

    n_phys = a.pods * a.pod_ch
    C = a.channels if a.channels is not None else n_phys
    dev_ch = a.cfg_dev_ch if a.cfg_dev_ch is not None else a.pod_ch
    stuck, quiet, leadoff = channel_list(a.stuck), channel_list(a.quiet), channel_list(a.leadoff)
    n_frames = STRIDES * 98 + 2
    if a.pods < 1 or a.pod_ch not in (4, 6, 8):
        sys.exit("--pods must be at least 1 and --pod-ch 8, 6 or 4 (ADS1299, ADS1299-6, ADS1299-4)")
    if not 1 <= C <= 24:
        sys.exit(f"C = {C}: the chip reads 1..24 channels (CMAX = 24)")
    if not 0 <= dev_ch <= 15 or not 0 <= a.shift <= 31:
        sys.exit("--cfg-dev-ch must be 0..15 (4-bit register) and --shift 0..31 (5-bit register)")
    if set(stuck) & set(quiet):
        sys.exit("a channel cannot be both --stuck and --quiet")
    for c in stuck + quiet:
        if not 0 <= c < min(C, n_phys):
            sys.exit(f"--stuck / --quiet channel {c}: must be an electrode the chip reads (0..{min(C, n_phys) - 1})")
    for c in leadoff:
        if not 0 <= c < n_phys:
            sys.exit(f"--leadoff channel {c}: must be an electrode that exists (0..{n_phys - 1})")
    if a.leadoff_frames:
        r = frame_range(a.leadoff_frames)
        if not leadoff or r.start < 0 or r.stop > n_frames:
            sys.exit(f"--leadoff-frames needs --leadoff and a range inside 0-{n_frames - 1}")
    if a.drop_frame >= 0 and not 98 <= a.drop_frame < STRIDES * 98 - 98:
        sys.exit(f"--drop-frame must be in 98..{STRIDES * 98 - 99}, so that a decision follows it")
    if C > n_phys:
        print(f"NOTE: C = {C} but the chain has only {n_phys} electrodes: channels {n_phys}..{C - 1} are missing")

    tb_dir = Path(__file__).resolve().parent
    rtl_dir = a.rtl_dir.resolve()
    sources = sorted(rtl_dir.glob("*.v")) + [tb_dir / "ads1299_model.v", tb_dir / "tb_chain.v"]
    params = {"NPOD": a.pods, "PCH": a.pod_ch, "DRDY_MISSING": int(a.no_drdy)}
    for name in ("t_conv", "t_h", "t_d", "t_csdod", "t_csdoz"):
        val = getattr(a, name)
        if val is not None:
            params[name] = float(val)
    if a.late_drdy:
        params["t_conv"] = LATE_T_CONV_NS

    name = f"{a.pods}x{a.pod_ch}_C{C}" + "".join(f"_{n}{v:g}" for n, v in params.items() if n.startswith("t_"))
    for label, val in (("cfg", a.cfg_dev_ch), ("sh", a.shift if a.shift != SHIFT else None),
                       ("drop", a.drop_frame if a.drop_frame >= 0 else None),
                       ("gap", a.cmd_gap if a.cmd_gap != 10.0 else None)):
        if val is not None:
            name += f"_{label}{val:g}"
    for label, lst in (("stuck", stuck), ("quiet", quiet), ("lo", leadoff)):
        if lst:
            name += f"_{label}" + "-".join(map(str, lst))
    if a.leadoff_frames:
        name += "_lof" + a.leadoff_frames
    if a.no_drdy:
        name += "_nodrdy"
    if a.late_drdy:
        name += "_late"
    if a.bad_c:
        name += "_badc"
    build = tb_dir / "sim_build_chain" / name
    build.mkdir(parents=True, exist_ok=True)
    print("RTL:", ", ".join(s.name for s in sources))
    print(f"Chain: {a.pods} ADS1299 device(s) x {a.pod_ch} electrodes = {n_phys} electrodes, chip reads C = {C}")

    build_args = []
    runner = Icarus()
    if a.waves:
        class IcarusVcd(Icarus):
            """The stock runner passes -none to vvp when waves=False, which suppresses $dumpfile."""
            def _test_command(self):
                return [["-vcd" if x == "-none" else x for x in cmd] for cmd in super()._test_command()]
        runner = IcarusVcd()
        wave_dir = rtl_dir / "waveform"
        wave_dir.mkdir(exist_ok=True)
        vcd = wave_dir / f"emg_chain_{name}.vcd"
        dump_v = build / "wave_dump.v"
        dump_v.write_text(f'module wave_dump();\n  initial begin\n    $dumpfile("{vcd.as_posix()}");\n'
                          f'    $dumpvars(0, tb);\n  end\nendmodule\n')
        sources.append(dump_v)
        build_args = ["-s", "wave_dump"]
        print("VCD:", vcd)

    runner.build(sources=sources, hdl_toplevel="tb", build_dir=build, always=True,
                 timescale=("1ns", "1ps"), parameters=params, build_args=build_args)
    results = runner.test(hdl_toplevel="tb", test_module="test_emg_chain", test_dir=tb_dir, build_dir=build,
                          extra_env={"EMG_CHANNELS": str(C), "EMG_PODS": str(a.pods), "EMG_POD_CH": str(a.pod_ch),
                                     "EMG_CFG_DEV_CH": str(dev_ch), "EMG_SHIFT": str(a.shift),
                                     "EMG_STUCK": a.stuck, "EMG_QUIET": a.quiet, "EMG_LEADOFF": a.leadoff,
                                     "EMG_LO_FRAMES": a.leadoff_frames, "EMG_DROP": str(a.drop_frame),
                                     "EMG_NO_DRDY": str(int(a.no_drdy)), "EMG_LATE": str(int(a.late_drdy)),
                                     "EMG_CMD_GAP": str(a.cmd_gap), "EMG_BAD_C": str(int(a.bad_c))})
    _, failed = get_results(Path(results))
    sys.exit(1 if failed else 0)
