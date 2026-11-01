"""
lda_golden_model.py - COMPATIBILITY LAYER over golden_model.py, legacy preset.

The RTL testbenches (test_emg_top.py, test_emg_chain.py) import these names. They now come
from golden_model.py with make_config("legacy") = the current RTL, so there is only one model.
Outputs are identical to the old lda_golden_model.py (checked by check_compat.py).

Names kept: LdaCoreRef, window_features, SUB_WIN, N_CLASSES, N_TYPES, BANK_BITS, ads_code_to_int8,
calibrate_norm, dataset_features, fit_int8_lda, quantize_lda, normalize, to_adc_int8,
mitchell_sq, POW_LUT, sample_features, INT32_MIN, INT32_MAX.
For the target preset, use golden_model.py directly (see README, "Testbench changes").
"""
import os
from dataclasses import replace

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import golden_model as GM

# RTL step 1 (31 Oct 2026): 1 kHz sampling, bank size B, bank count N and warm-up are settings
# (config 0x7F5, 0x7F6, 0x7F7). The testbenches read them from these environment variables;
# defaults = the new design's window (3 banks of 50) with a short warm-up to keep runs fast.
# EMG_BANK_LEN=98 EMG_N_BANKS=2 EMG_WARMUP=1 gives the old chip's behaviour.
# RTL step 2 (1 Nov 2026): 24-bit mu', signed 6-bit k (-32..31). EMG_REQUANT=old gives the old
# calibration rule (k >= 1, 22-bit mu), which the step-2 chip still runs exactly.
_OLD_RQ = os.environ.get("EMG_REQUANT", "new") == "old"
CFG = GM.make_config("legacy", fs_hz=1000.0,
                     bank_len=int(os.environ.get("EMG_BANK_LEN", 50)),
                     n_banks=int(os.environ.get("EMG_N_BANKS", 3)),
                     warmup=int(os.environ.get("EMG_WARMUP", 2)),
                     k_min=1 if _OLD_RQ else -32, k_max=22 if _OLD_RQ else 31,
                     mu_bits=22 if _OLD_RQ else 24,
                     # RTL step 3 (1 Nov 2026): no LDA on the chip. EMG_CHIP_LDA=1 keeps the old
                     # 6-class Stage 3 in the model (check_compat.py uses it).
                     chip_classes=6 if os.environ.get("EMG_CHIP_LDA") == "1" else 0,
                     # RTL step 4 (1 Nov 2026): rest flag (rest_sum < T_REST) and the dead check on
                     # the 24-bit codes. EMG_DEAD_RULE=wl_int8 keeps the old dead test (WL window
                     # sum of the 8-bit values = 0; check_compat.py uses it).
                     rest_gate=True,
                     dead_rule=os.environ.get("EMG_DEAD_RULE", "code24"))
# RTL step 5 (1 Nov 2026): the input filter (setting 0x7FC = 1). FILTER_DESIGN = the designed filter;
# its 17 coefficients (FILTER_COEFS, chip order) are the chip's reset values (0x7C0 - 0x7D0).
FILTER_DESIGN = GM.design_filter(replace(CFG, input_path="filter"))
FILTER_COEFS = ([v for s in FILTER_DESIGN.sections[:4]
                 for v in (s["g"] if s["g"] is not None else 1 << FILTER_DESIGN.F, s["a1"], s["a2"])]
                + [FILTER_DESIGN.sections[4][n] for n in ("b0", "b1", "b2", "a1", "a2")])


def filter_int8(codes, shift):
    """Whole recordings (T x C 24-bit codes) through the chip's filter -> T x C INT8 (first-sample start)."""
    y = GM.filter_wide(GM.code_to_filter_input(codes, CFG), FILTER_DESIGN)
    return GM.filter_out_int8(y, shift, FILTER_DESIGN.S)


def filter_shift_for(recordings, skip=0):
    """Output shift 0..8 from training recordings (list of T x C codes): the smallest shift that keeps
    the loudest filter output (after `skip` samples of each recording) inside -128..127."""
    m = max(int(np.abs(GM.filter_wide(GM.code_to_filter_input(r, CFG), FILTER_DESIGN)[skip:]).max())
            for r in recordings)
    return GM.choose_filter_shift(m, FILTER_DESIGN.S, CFG.filt_out_shift_max)


N_CLASSES = 6                           # classes of the old on-chip LDA (until RTL step 2)   [rtl]
N_TYPES = GM.N_TYPES
SUB_WIN = CFG.bank_len                  # samples per bank = stride; window = N banks        [rtl]
N_BANKS = CFG.n_banks
WARMUP = CFG.warmup                     # read-outs not sent out after reset
BANK_BITS = GM.BANK_BITS
INT32_MIN, INT32_MAX = GM.INT32_MIN, GM.INT32_MAX
POW_LUT = GM.POW_LUT["one"]


def to_adc_int8(emg, full_scale):
    """Old 8-bit ADC model (synthetic tests only): electrode voltage -> INT8."""
    return np.clip(np.round(emg / full_scale * 128), -128, 127).astype(np.int8)


def ads_code_to_int8(code, shift):
    """ADS1299 24-bit code -> INT8 as the legacy chip does it: clip(code >>> shift).
    The RTL treats a shift above 16 as 16; so does this function (the old one did not clamp,
    which only differs for shifts the RTL never uses)."""
    return GM.ads_code_to_int8(code, shift).astype(np.int8)


def mitchell_sq(a):
    return GM.mitchell_sq(a, "one")


def sample_features(x, xp, d0):
    """Stage 1 for one channel: (SABS, WL, ZC, SSC, POW) and d1 = x - xp."""
    a, wl, zc, ssc, pw, d1 = GM.per_sample_features(np.int64(x), np.int64(xp), np.int64(d0))
    return (int(a), int(wl), int(zc), int(ssc), int(pw)), int(d1)


def normalize(raw, mu, k):
    """Shift normalizer, k >= 1 in the legacy chip: clip(((raw - mu) + 2^(k-1)) >> k)."""
    return GM.requant(raw, mu, k)


class LdaCoreRef:
    """Sample-by-sample chip (Stages 1-3). Outputs every SUB_WIN samples:
    dict(cls, scores, q, full, valid, dead, rest, rest_sum, peak_cycles); valid = read-out sent out
    (past warm-up). Step 4: t_rest = the T_REST setting (None or 0: rest never set), dead_low8 = the
    dead rule setting (0x7F9), and step() needs the 24-bit codes the chip received for the dead check.
    Step 5: filt_shift = the filter's output shift (setting 0x7F3) turns the filter on (setting
    0x7FC = 1): step() then takes the codes only (sample is ignored) and last_x8 holds the filter's
    8-bit outputs for that sample."""

    def __init__(self, C, mu, k, W=None, b=None, t_rest=None, dead_low8=False, filt_shift=None):
        self.C, self.D = C, N_TYPES * C
        cfg = CFG
        if dead_low8:
            assert CFG.dead_rule == "code24", "dead_low8 needs the 24-bit dead check"
            cfg = replace(cfg, dead_rule="code_low8")
        self.filt = filt_shift is not None
        if self.filt:
            cfg = replace(cfg, input_path="filter")
        self.ref = GM.ChipRef(cfg, C, mu, k, filt_shift=filt_shift, t_rest=t_rest, W=W, b=b)
        self.last_x8 = None
        self.strides = 0
        self.cycles = 0

    def step(self, sample, codes=None):
        start = self.cycles
        if self.filt:
            self.last_x8, out = self.ref.step([int(c) for c in codes])
        else:
            out = self.ref.step_int8([int(v) for v in sample], None if codes is None else [int(c) for c in codes])
        self.cycles += self.C                                     # Stage 1: one channel per cycle
        if out is None:
            return None
        self.strides += 1
        self.cycles += self.D                                     # Stage 2
        if CFG.chip_classes:
            self.cycles += N_CLASSES * (self.D + 1) + 1           # Stage 3 (old chip only)
        res = dict(q=out["q"], full=self.strides >= CFG.n_banks, valid=out["valid"], dead=out["dead"],
                   rest=bool(out["rest"]), rest_sum=int(out["rest_sum"]), peak_cycles=self.cycles - start)
        if CFG.chip_classes:                                      # the old on-chip LDA
            res.update(cls=out["cls"], scores=out["scores"])
        return res


def window_features(rec, C):
    """Stages 1-2 for a whole recording (T x C INT8): raw window sums (n x D) and full-window flags."""
    f = GM.chip_features(np.asarray(rec, np.int64)[:, :C], replace(CFG, dead_rule="none"))
    return GM.window_matrix(f["win"])[:, 0, :], np.arange(f["win"].shape[0]) >= CFG.n_banks - 1


def calibrate_norm(raw):
    """mu = rounded mean; k = smallest shift that never clips on the calibration set.
    Step 2: k may be <= 0 (stretch) and mu has 24 bits; EMG_REQUANT=old keeps k >= 1, 22-bit mu."""
    mu, k = GM.calibrate_requant(raw, CFG, GM.rule_max)
    assert mu.max() < 2 ** CFG.mu_bits and CFG.k_min <= k.min() and k.max() <= CFG.k_max
    return mu, k


def pack_norm(mu, k):
    """Config word for addresses 0x000-0x077 (step 2): bits 23:0 = mu', bits 29:24 = k (signed 6 bits)."""
    assert 0 <= int(mu) < (1 << 24) and -32 <= int(k) <= 31
    return ((int(k) & 0x3F) << 24) | int(mu)


def quantize_lda(coef, intercept):
    """Legacy chip LDA: INT8 weights with ONE scale s_w, INT32 bias on the same scale; unused
    classes get zero weights and bias INT32_MIN.                                       [rtl][lgm]"""
    n = len(intercept)
    assert coef.shape[0] == n and 3 <= n <= N_CLASSES
    s_w = np.abs(coef).max() / 127
    W = np.round(coef / s_w).astype(np.int64)
    b = np.round(intercept / s_w).astype(np.int64)
    assert np.abs(b).max() <= INT32_MAX
    W = np.vstack([W, np.zeros((N_CLASSES - n, W.shape[1]), np.int64)])
    b = np.concatenate([b, np.full(N_CLASSES - n, INT32_MIN, np.int64)])
    return W, b


def fit_int8_lda(q, y):
    lda = LinearDiscriminantAnalysis().fit(q, y)
    return (lda, *quantize_lda(lda.coef_, lda.intercept_))


def dataset_features(recordings, C):
    """Full windows only, from a list of (INT8 recording T x C, label)."""
    R, Y = [], []
    for rec, label in recordings:
        raw, full = window_features(rec, C)
        R.append(raw[full])
        Y += [label] * int(full.sum())
    return np.vstack(R), np.array(Y)
