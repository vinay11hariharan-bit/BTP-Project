"""
Unit test for emg_shift_norm (Stage 2) alone: bank sums -> 8-bit features, against the golden
model's requant(). Step 2 (1 Nov 2026): 24-bit mu', signed 6-bit k (-32..31).

Each trial: random channel count C, random bank words for 4 banks with a random set of banks in
use, random mu' (all 24 bits) and k (every value -32..31 appears), plus hand-picked edge cases
where the result is exactly at, just inside or just outside -128..127. Checks every x_q value,
the dead-channel flags (step 4: dead = channel < C and its chg_win bit is 0, from a random
chg_win) and the rest sum (step 4: the SABS window sums of the C channels added up).

Run (from the RTL folder):   python tb/test_shift_norm.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_model as GM                                   # noqa: E402

FIELDS = ((0, 14), (14, 15), (29, 7), (36, 7), (43, 21))   # (lsb, width): SABS WL ZC SSC POW
TRIALS = int(os.environ.get("EMG_TRIALS", 300))


def pack(fields):
    w = 0
    for (lsb, wid), v in zip(FIELDS, fields):
        assert 0 <= v < (1 << wid)
        w |= int(v) << lsb
    return w


def make_trial(rng, t):
    C = int(rng.integers(1, 25))
    D = 5 * C
    use = int(rng.integers(1, 16))                         # banks in the window
    banks = np.zeros((4, C, 5), np.int64)
    for b in range(4):
        for i, (_, wid) in enumerate(FIELDS):
            hi = (1 << wid) if rng.random() < 0.3 else min(1 << wid, 2000 if i < 4 else 400000)
            banks[b, :, i] = rng.integers(0, hi, size=C)
    raw = sum(banks[b] for b in range(4) if use >> b & 1).reshape(D)
    k = rng.integers(-32, 32, size=D)
    k[: min(D, 64)] = np.arange(-32, 32)[: min(D, 64)]     # every k value appears
    rng.shuffle(k)
    mu = np.where(rng.random(D) < 0.2, rng.integers(0, 1 << 24, size=D),
                  np.clip(raw + rng.integers(-300, 300, size=D) * (2 ** np.clip(k, 0, 20)), 0, (1 << 24) - 1))
    # edge cases: d chosen so that the result is exactly 127 / 128 / -128 / -129 before clipping
    for f in range(0, D, 3):
        kk = int(k[f])
        target = int(rng.choice([127, 128, -128, -129, 0, -1, 1]))
        d = target << kk if kk > 0 else (target >> -kk if target >= 0 else -((-target) >> -kk))
        if kk > 0 and rng.random() < 0.5:
            d += int(rng.integers(-(1 << (kk - 1)), 1 << (kk - 1)))   # inside the rounding step
        m = int(raw[f]) - d
        if 0 <= m < (1 << 24):
            mu[f] = m
    return C, use, banks, mu.astype(np.int64), k.astype(np.int64), raw


async def ram_model(dut, words):
    """emg_feat_accum's bank RAMs: 1-cycle read latency, output held while s2_re is low."""
    q = [0, 0, 0, 0]
    while True:
        await RisingEdge(dut.clk)
        re, addr = int(dut.s2_re.value), int(dut.s2_addr.value)
        if re:
            q = [words[b][addr] if addr < len(words[b]) else 0 for b in range(4)]
        dut.s2_q0.value, dut.s2_q1.value, dut.s2_q2.value, dut.s2_q3.value = q


async def norm_model(dut, mu, k):
    """emg_cfg_regs' norm memory: 1-cycle read latency."""
    while True:
        await RisingEdge(dut.clk)
        if int(dut.norm_re.value):
            f = int(dut.norm_addr.value)
            dut.norm_mu.value = int(mu[f])
            dut.norm_k.value = int(k[f]) & 0x3F


@cocotb.test()
async def requant(dut):
    cocotb.start_soon(Clock(dut.clk, 1000, unit="ns").start())
    rng = np.random.default_rng(int(os.environ.get("EMG_SEED", 2)))
    dut.rst_n.value = 0
    dut.start.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    words, mu_k = [[], [], [], []], [np.zeros(1), np.zeros(1)]
    cocotb.start_soon(ram_model(dut, words))
    n_checked = n_left = n_clip = 0
    for t in range(TRIALS):
        C, use, banks, mu, k, raw = make_trial(rng, t)
        D = 5 * C
        for b in range(4):
            words[b][:] = [pack(banks[b, c]) for c in range(C)]
        await FallingEdge(dut.clk)
        dut.cfg_n_ch.value = C
        dut.bank_use.value = use
        chg = int(rng.integers(0, 1 << 24)) if t % 3 else (1 << 24) - 1 - (1 << int(rng.integers(0, C)))
        dut.chg_win.value = chg                             # (every third trial: one dead channel)
        nm = cocotb.start_soon(norm_model(dut, mu, k))
        dut.start.value = 1
        await FallingEdge(dut.clk)
        dut.start.value = 0
        got = {}
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.xq_we.value):
                got[int(dut.xq_addr.value)] = dut.xq_data.value.to_signed()
            if int(dut.done.value):
                break
        nm.cancel()
        want = GM.requant(raw, mu, k)
        assert sorted(got) == list(range(D)), f"trial {t}: x_q addresses {sorted(got)[:8]}..."
        for f in range(D):
            assert got[f] == want[f], (f"trial {t} (C={C}, banks used {use:04b}): x_q[{f}] = {got[f]}, expected "
                                       f"{want[f]}  (raw {raw[f]}, mu' {mu[f]}, k {k[f]})")
        want_dead = sum(1 << c for c in range(C) if not chg >> c & 1)
        assert int(dut.dead_mask.value) == want_dead, (f"trial {t}: dead_mask {int(dut.dead_mask.value):06x}, "
                                                       f"expected {want_dead:06x} (chg_win {chg:06x}, C={C})")
        want_rest = int(sum(raw[5 * c] for c in range(C)))
        assert int(dut.rest_sum.value) == want_rest, f"trial {t}: rest_sum {int(dut.rest_sum.value)}, expected {want_rest}"
        n_checked += D
        n_left += int(np.sum(k <= 0))
        n_clip += int(np.sum((want == 127) | (want == -128)))
    dut._log.info(f"{TRIALS} trials, {n_checked} features match requant(): {n_left} with k <= 0 (stretch), "
                  f"{n_clip} clipped at -128 / 127, every k from -32 to 31, mu' over all 24 bits; dead mask and rest sum "
                  f"match in every trial")


if __name__ == "__main__":
    from cocotb_tools.runner import Icarus
    rtl = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    runner = Icarus()
    build = Path(__file__).resolve().parent / "sim_build_norm"
    runner.build(sources=[rtl / "emg_shift_norm.v"], hdl_toplevel="emg_shift_norm", build_dir=build,
                 always=True, timescale=("1ns", "1ps"))
    runner.test(hdl_toplevel="emg_shift_norm", test_module="test_shift_norm", test_dir=Path(__file__).parent,
                build_dir=build)
