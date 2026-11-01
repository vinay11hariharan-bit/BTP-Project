"""
Unit test for emg_filter (step 5, 1 Nov 2026) alone, against the golden model (golden_model.py,
ChipRef._filter with the "target" settings: 5 sections, 14-bit coefficients, S = 4 stored
fractional bits, first-sample start).

After every sample it checks, for every channel:
  * the 8-bit output x8 (every output shift 0..8 is used)
  * the whole memory word written back: input history and BOTH stored outputs of all 5 sections
    (so an internal error shows even when the 8-bit output clips)
Inputs:
  random       random full-scale 16-bit inputs, random C
  offset       a large offset (the electrode's DC) plus 50 Hz hum and noise: with the first-sample
               start the output must stay small from the very first sample (no jump)
  worst        the input that drives the notch output to its largest possible value (the sign of
               the whole filter's impulse response, time-reversed, at full scale), and its negative:
               the largest stored value seen must reach the bound of golden_model.filter_widths
               and never overflow
  re-arm       enable = 0 between runs: the next sample is a first sample again

Run (from the RTL folder):   python tb/test_filter.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_model as GM                                   # noqa: E402

CMAX = 24
CFG = GM.make_config("target")
DESIGN = GM.design_filter(CFG)


def coef_list(design):
    """The 17 coefficients in the chip's order: per section 0-3 g, a1, a2; notch b0 b1 b2 a1 a2."""
    out = []
    for s in design.sections[:4]:
        assert s["kind"] == "shift"
        out += [s["g"] if s["g"] is not None else 1 << design.F, s["a1"], s["a2"]]
    n = design.sections[4]
    assert n["kind"] == "gen"
    return out + [n["b0"], n["b1"], n["b2"], n["a1"], n["a2"]]


def coef_flat(design):
    return sum((c & 0xFFFF) << (16 * i) for i, c in enumerate(coef_list(design)))


def signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def unpack_word(w):
    """RAM word -> (x16[n-1], x16[n-2], [(y1, y2) for the 5 sections])."""
    u1, u2 = signed(w, 16), signed(w >> 16, 16)
    ys = [(signed(w >> (32 + 44 * k), 22), signed(w >> (54 + 44 * k), 22)) for k in range(4)]
    ys.append((signed(w >> 208, 23), signed(w >> 231, 23)))
    return u1, u2, ys


def worst_input(n):
    """Full-scale input whose last n samples drive the notch output to its largest value."""
    imp = np.zeros(n); imp[0] = 1.0
    h = imp
    for s in DESIGN.sections:
        if s["kind"] == "shift":
            g = (s["g"] if s["g"] is not None else 1 << DESIGN.F) / (1 << DESIGN.F)
            b = np.array([1.0, 2.0 * s["sign"], 1.0]) * g
        else:
            b = np.array([s["b0"], s["b1"], s["b2"]]) / (1 << DESIGN.F)
        a = np.array([1.0, s["a1"] / (1 << DESIGN.F), s["a2"] / (1 << DESIGN.F)])
        h = lfilter(b, a, h)
    return np.where(h[::-1] >= 0, 32767, -32768).astype(np.int64)


async def run_block(dut, x16, shift, tag, stats):
    """x16: (T, C) inputs. Re-arms the filter, then runs T samples; checks every sample."""
    T, C = x16.shape
    ref = GM.ChipRef(CFG, C, np.zeros(5 * C), np.zeros(5 * C), filt_shift=shift)
    await FallingEdge(dut.clk)
    dut.enable.value = 0                                   # re-arm: next sample is a first sample
    dut.cfg_n_ch.value = C
    dut.cfg_shift.value = shift
    await FallingEdge(dut.clk)
    dut.enable.value = 1
    for t in range(T):
        await FallingEdge(dut.clk)
        dut.rx16_flat.value = sum((int(v) & 0xFFFF) << (16 * c) for c, v in enumerate(x16[t]))
        dut.start.value = 1
        await FallingEdge(dut.clk)
        dut.start.value = 0
        words, cycles = {}, 1
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            cycles += 1
            if int(dut.u_mem.we.value):
                words[int(dut.u_mem.waddr.value)] = int(dut.u_mem.wdata.value)
            if int(dut.done.value):
                break
        stats["cycles"] = max(stats["cycles"], cycles)
        x8 = int(dut.x8_flat.value)
        for c in range(C):
            want8 = ref._filter(c, int(x16[t, c]))
            got8 = signed(x8 >> (8 * c), 8)
            u1, u2, ys = unpack_word(words[c])
            want_u = [v >> DESIGN.S for v in ref.u[c]]
            want_ys = [tuple(p) for p in ref.ys[c]]
            assert (u1, u2) == tuple(want_u), f"{tag} sample {t} ch {c}: input history {(u1, u2)}, expected {want_u}"
            for k in range(5):
                assert ys[k] == want_ys[k], (f"{tag} sample {t} ch {c}: section {k} memory {ys[k]}, "
                                             f"expected {want_ys[k]}")
                stats["ymax"][k] = max(stats["ymax"][k], abs(ys[k][0]))
            assert got8 == want8, f"{tag} sample {t} ch {c} (shift {shift}): x8 = {got8}, expected {want8}"
            stats["n"] += 1
            stats["clip"] += int(want8 in (-128, 127))


@cocotb.test()
async def filt(dut):
    cocotb.start_soon(Clock(dut.clk, 1000, unit="ns").start())
    rng = np.random.default_rng(int(os.environ.get("EMG_SEED", 5)))
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.enable.value = 0
    dut.coef_flat.value = coef_flat(DESIGN)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    stats = dict(n=0, clip=0, cycles=0, ymax=[0] * 5)

    # random full-scale inputs, every output shift
    for shift in range(9):
        C = int(rng.integers(1, CMAX + 1)) if shift else CMAX
        x16 = rng.integers(-32768, 32768, size=(12, C))
        await run_block(dut, x16, shift, f"random C={C}", stats)
    # a large offset + 50 Hz hum + noise: no jump at the first sample
    C, T = 4, 120
    t = np.arange(T)[:, None]
    off = np.array([[20000, -25000, 9000, -500]])
    x16 = (off + 300 * np.sin(2 * np.pi * 50 * t / 1000 + np.arange(C)) + rng.normal(0, 40, (T, C))).astype(np.int64)
    n0 = stats["n"]
    ref = GM.ChipRef(CFG, C, np.zeros(5 * C), np.zeros(5 * C), filt_shift=4)
    first_out = [ref._filter(c, int(x16[0, c])) for c in range(C)]
    assert first_out == [0] * C, f"golden model: first output {first_out} with the first-sample start"
    await run_block(dut, x16, 4, "offset", stats)
    # worst case: drives the notch output to its bound, both signs
    w = worst_input(1500)
    x16 = np.stack([w, -w - 1], axis=1)            # -w - 1 keeps -32768 .. 32767
    await run_block(dut, x16, 8, "worst", stats)
    bound = [r["state_max_bound"] for r in GM.filter_widths(DESIGN)]
    dut._log.info(f"largest stored output per section {stats['ymax']} (bounds {bound}; widths 22, 22, 22, 22, 23 bits)")
    assert all(m <= b for m, b in zip(stats["ymax"], bound))
    assert stats["ymax"][4] >= 0.9 * bound[4], "the worst-case input did not reach the notch's bound"
    dut._log.info(f"{stats['n']} channel-samples match the golden model (8-bit output, input history and both "
                  f"stored outputs of all 5 sections), {stats['clip']} clipped at -128 / 127, every shift 0..8, "
                  f"first-sample start, re-arm; longest run {stats['cycles']} cycles")


if __name__ == "__main__":
    from cocotb_tools.runner import Icarus, get_results
    rtl = Path(__file__).resolve().parents[1]
    build = Path(__file__).resolve().parent / "sim_build_filter"
    runner = Icarus()
    runner.build(sources=[rtl / "emg_filter.v", rtl / "emg_tp_ram.v"], hdl_toplevel="emg_filter",
                 build_dir=build, always=True, timescale=("1ns", "1ps"))
    results = runner.test(hdl_toplevel="emg_filter", test_module="test_filter", test_dir=Path(__file__).parent,
                          build_dir=build)
    _, failed = get_results(Path(results))
    sys.exit(1 if failed else 0)
