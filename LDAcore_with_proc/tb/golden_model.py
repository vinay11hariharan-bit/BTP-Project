"""
golden_model.py - ONE bit-exact model of the sEMG chip (feature extraction) and of the
processor (INT8 LDA, classifier route (a)).

Used for three jobs that must agree bit for bit:
  1. test vectors for the Verilog RTL (make_vectors.py, cocotb testbenches)
  2. training the factory model (fors_chip_study.py)
  3. calibration on the host (calibrate_requant, calibrate_t_rest, choose_filter_shift)

Where every rule comes from (cited next to the code as [source]):
  [plan]      EMG_implementation_plan.pdf (30 Oct 2026) and the decisions in the request:
              route (a), filter input = top 16 bits, gain 24
  [spec n]    the request "THE AGREED DESIGN", item n
  [hw]        fors_emg_hw_bitexact.py (chip_sos, fixed_point_sections, chip_pass,
              mitchell_table, ShiftNormalizer, calibrate)
  [lgm]       RTL/tb/lda_golden_model.py (ADS1299 version): structure, names, legacy rules
  [rtl]       the current RTL (emg_spi_adc, emg_feat_accum, emg_feat_alu, emg_mitchell_sq,
              emg_shift_norm, emg_lda_mac, emg_core_ctrl)
  [win]       window_test.py (classifier pipeline, 150/50 setting)
  [rest]      rest_test.py (Rest class, strength gate)
  [review]    RTL Review: Feature Extraction and LDA Core vs Agreed Design (PDF)
Open questions are marked OPEN-Qn and listed in README.md.

Two presets (make_config("legacy") / make_config("target")):
  legacy = the RTL as it is today: 985 Hz, 2 banks x 98, no filter,
           INT8 = clip(code >>> shift), k >= 1, one-segment Mitchell, strict SSC,
           6-class LDA on the chip, dead flag = WL window sum 0, first read-out masked.
  target = the agreed design: 1 kHz, ADS1299 code -> top 16 bits -> 5-biquad fixed-point
           filter -> shift 0..8 -> INT8, 3 banks x 50, signed k, 24-bit mu', rest gate,
           dead flag on the 24-bit code, 10 read-outs of warm-up, features to the processor.
Every setting is an independent field, so a later RTL step can flip one at a time.

Integer path: numpy int64 / Python int only, no floats (except the ADC model, which turns
volts into codes, and the one-off coefficient design).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, lfilter, resample_poly

# =====================================================================================
# Constants
# =====================================================================================
N_TYPES = 5                                  # per channel: SABS, WL, ZC, SSC, POW   [lgm]
TYPE_NAMES = ("SABS", "WL", "ZC", "SSC", "POW")
BANK_BITS = (14, 15, 7, 7, 21)               # bank word fields, 64 bits             [rtl][spec 6]
INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -2 ** 31, 2 ** 31 - 1
CMAX = 24                                    # largest channel count of the chip     [rtl]
CODE_MIN, CODE_MAX = -(1 << 23), (1 << 23) - 1   # ADS1299 24-bit two's complement   [spec 2]


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass(frozen=True)
class ChipConfig:
    name: str
    # --- sampling                                                                   [spec 1]
    fs_hz: float                  # chip sample rate (legacy 8 MHz / 8120 = 985.2 Hz; target 1000)
    # --- ADC: TI ADS1299                                                            [spec 2][plan]
    gain: int = 24
    vref: float = 4.5
    adc_model: str = "ideal"      # "ideal" | "sinc3_single8k" | "sinc3_cont2k"  (OPEN-Q5)
    # --- input path
    input_path: str = "filter"    # "filter" (target) | "direct" (legacy: INT8 = clip(code >>> int8_shift))
    int8_shift: int = 11          # legacy direct path, 0..16 (RTL clamps above 16)    [rtl]
    filt_in_shift: int = 8        # filter input = code >> 8 = top 16 bits             [spec 3][plan]
    coef_frac: int = 14           # 16-bit coefficients, 14 fractional bits            [spec 4][hw]
    coef_bits: int = 16
    bandpass_hz: tuple = (20.0, 450.0)                                                # [spec 4][hw]
    bandpass_order: int = 4       # butter(4, ...) bandpass -> 4 biquads              # [hw]
    notch_hz: float = 50.0
    notch_q: float = 30.0         # [hw] via fors_emg_lda_baseline.NOTCH_Q
    filt_out_shift_max: int = 8   # output right shift 0..8                            [spec 4]
    state_frac: int = 4           # bits kept after the binary point in the filter's input history and
                                  # section memory (S). Decided 31 Oct: 4 (memory 22-23 bits per section).
                                  # FORS-EMG study: 8, 6, 4 cost nothing against 14 (memory 32-33 bits);
                                  # 2 lost 0.24 points (calibrated) and 0 lost 1.8, so 4 is the smallest safe
    filter_init: str = "first"    # filter memory after reset: "first" = both input-history registers
                                  # take the first sample (decided 31 Oct: a 100 mV electrode offset
                                  # otherwise arrives as a jump and the notch rings for ~1 s, costing
                                  # ~1 point on FORS-EMG); "zero" = all memory 0 (the spec's rule)
    # --- per-sample features                                                         [spec 5]
    ssc_rule: str = "strict"      # strict two-bit rule (RTL, B3)                       [review D1]
    mitchell: str = "one"         # "one" | "two"                                      [review D4]
    # --- banks                                                                       [spec 6]
    bank_len: int = 50            # B, 1..100
    n_banks: int = 3              # N, 1..4
    # --- requantizer                                                                 [spec 7]
    k_min: int = -32              # signed 6-bit k
    k_max: int = 31
    mu_bits: int = 24             # mu' width (unsigned: window sums are never negative)
    # --- rest gate                                                                   [spec 8]
    rest_gate: bool = True
    # --- dead-channel flag                                                           [spec 9]
    dead_rule: str = "code24"     # "code24" | "code_low8" | "wl_int8" (legacy RTL) | "none"
    # --- warm-up                                                                     [spec 10]
    warmup: int = 10              # read-outs discarded after reset
    # --- classifier on the chip (legacy only)                                        [rtl]
    chip_classes: int = 0         # 6 on the legacy chip (Stage 3), 0 = features go to the processor

    @property
    def window_samples(self) -> int:
        return self.bank_len * self.n_banks

    def validate(self):
        assert self.input_path in ("filter", "direct")
        assert self.adc_model in ("ideal", "sinc3_single8k", "sinc3_cont2k")
        assert self.mitchell in ("one", "two") and self.ssc_rule == "strict"
        assert self.filter_init in ("first", "zero")
        assert 0 <= self.state_frac <= self.coef_frac
        assert 1 <= self.bank_len <= 100 and 1 <= self.n_banks <= 4, "WINDOW register range [spec 6]"
        assert self.dead_rule in ("code24", "code_low8", "wl_int8", "none")
        assert 0 <= self.int8_shift <= 31
        # R4 [review]: the warm-up must cover the read-outs whose window still holds banks from
        # before reset (uncleared RAM in the RTL)
        assert self.warmup >= self.n_banks - 1, "warm-up shorter than the banks per window (review R4)"
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def make_config(preset: str, **changes) -> ChipConfig:
    """'legacy' = the current RTL; 'target' = the agreed design. Any field can be overridden."""
    if preset == "legacy":
        cfg = ChipConfig(name="legacy", fs_hz=8e6 / 8120, input_path="direct", int8_shift=11,
                         bank_len=98, n_banks=2, k_min=1, k_max=22, mu_bits=22, rest_gate=False,
                         dead_rule="wl_int8", warmup=1, chip_classes=6, mitchell="one")
    elif preset == "target":
        cfg = ChipConfig(name="target", fs_hz=1000.0)
    else:
        raise ValueError(preset)
    return replace(cfg, **changes).validate() if changes else cfg.validate()


# =====================================================================================
# ADC model (ADS1299)                                                     [spec 2]
# =====================================================================================
def volts_to_codes(v, cfg: ChipConfig, dc_offset_v=None):
    """Electrode voltage (T x C, volts) -> 24-bit ADS1299 codes (int64).
    code = round(v * gain / vref * 2^23), clipped to 24 bits.            [spec 2]
    dc_offset_v: optional per-channel DC electrode offset added before the ADC (volts).
    Rounding is numpy rint (half to even); the ADC model is not part of the RTL."""
    v = np.asarray(v, dtype=np.float64)
    if dc_offset_v is not None:
        v = v + np.asarray(dc_offset_v, dtype=np.float64)[None, :]
    if cfg.adc_model != "ideal":
        v = sinc3_model(v, cfg)
    code = np.rint(v * cfg.gain / cfg.vref * (1 << 23))
    return np.clip(code, CODE_MIN, CODE_MAX).astype(np.int64)


def sinc3_model(v, cfg: ChipConfig):
    """OPTIONAL ADS1299 sinc3 decimation filter                            [spec 2] (OPEN-Q5)
    sinc3 impulse response = three boxcars of length 1/DR. The input is upsampled 64x
    (band-limited interpolation), convolved, and sampled at the chip's sample instants.
      sinc3_single8k : single-shot, DR = 8 kSPS, one conversion per START (box = 1/8000 s)
      sinc3_cont2k   : continuous DR = 2 kSPS, the chip keeps every 2nd sample (box = 1/2000 s).
                       The 'low-pass before the 2:1' is taken to be the sinc3 itself (OPEN-Q5).
    LIMIT: recordings sampled at ~1 kHz carry nothing above ~500 Hz, so this models only the
    in-band droop and delay of the sinc3, not aliasing from above 500 Hz.
    Implementation: for a band-limited input, sampling the sinc3 output at the chip instants is
    the same as filtering the 1 kHz samples with H(f) = sinc(f/DR)^3 * exp(-j 2 pi f 1.5/DR)
    (three boxcars of length 1/DR, total delay 1.5/DR), done here with an FFT."""
    dr = 8000.0 if cfg.adc_model == "sinc3_single8k" else 2000.0
    T = v.shape[0]
    n = 1 << int(np.ceil(np.log2(2 * T)))                 # zero padding: no wrap-around
    f = np.fft.rfftfreq(n, 1.0 / cfg.fs_hz)
    H = np.sinc(f / dr) ** 3 * np.exp(-2j * np.pi * f * 1.5 / dr)
    X = np.fft.rfft(v, n=n, axis=0)
    return np.fft.irfft(X * H.reshape((-1,) + (1,) * (v.ndim - 1)), n=n, axis=0)[:T]


def code_to_filter_input(code, cfg: ChipConfig):
    """Top 16 bits of each 24-bit code: code >> 8 (arithmetic).            [spec 3][plan]"""
    return np.asarray(code, dtype=np.int64) >> cfg.filt_in_shift


def ads_code_to_int8(code, shift):
    """Legacy direct path: clip(code >>> shift, -128, 127). Shift above 16 acts as 16.  [lgm][rtl]"""
    shift = min(int(shift), 16)
    return np.clip(np.asarray(code, dtype=np.int64) >> shift, INT8_MIN, INT8_MAX).astype(np.int64)


# =====================================================================================
# Filter: 5 biquads, Direct Form I, fixed point                           [spec 4][hw]
# =====================================================================================
@dataclass
class FilterDesign:
    sections: list          # integer sections, in chip order (dicts as in [hw] fixed_point_sections)
    sos: np.ndarray         # float sos in the same order (unquantized, for reference)
    order_note: str
    F: int
    init: str = "zero"      # "first": input history starts as the first sample (cfg.filter_init)
    S: int = 14             # fractional bits of the stored numbers (cfg.state_frac); F stays the coefficients' 14


def design_filter(cfg: ChipConfig) -> FilterDesign:
    """butter(4) band-pass 20-450 Hz + iirnotch 50 Hz Q 30 at the chip rate, as in [hw] chip_sos.
    Chip order: the high-pass sections first (numerator 1, -2, 1), then the low-pass sections
    (1, 2, 1), then the notch.                                             [spec 4]
    The gain scipy puts in the first band-pass section stays in that section."""
    fs = cfg.fs_hz
    bp = butter(cfg.bandpass_order, list(cfg.bandpass_hz), btype="bandpass", fs=fs, output="sos")
    nt = tf2sos(*iirnotch(cfg.notch_hz, cfg.notch_q, fs))
    F = cfg.coef_frac

    def shape(s):
        g = s[0]
        return s[:3] / g if g != 0 else s[:3]

    hp = [s for s in bp if np.allclose(shape(s), [1, -2, 1], atol=1e-9)]
    lp = [s for s in bp if np.allclose(shape(s), [1, 2, 1], atol=1e-9)]
    assert len(hp) + len(lp) == len(bp), "band-pass section with an unexpected numerator"
    sos = np.vstack(hp + lp + [nt[0]])
    lim = 1 << (cfg.coef_bits - 1)
    q = lambda v: int(np.round(v * (1 << F)))

    def chk(c):
        assert -lim <= c < lim, f"coefficient {c} does not fit {cfg.coef_bits} bits"
        return c

    secs = []
    for s in sos:
        b, a = s[:3], s[3:]
        g = b[0]
        sh = b / g if g != 0 else b
        if np.allclose(sh, [1, 2, 1], atol=1e-9) or np.allclose(sh, [1, -2, 1], atol=1e-9):
            secs.append({"kind": "shift", "sign": int(np.sign(sh[1])),
                         "g": None if abs(g - 1) < 1e-12 else chk(q(g)),
                         "a1": chk(q(a[1])), "a2": chk(q(a[2]))})
        else:
            secs.append({"kind": "gen", "b0": chk(q(b[0])), "b1": chk(q(b[1])), "b2": chk(q(b[2])),
                         "a1": chk(q(a[1])), "a2": chk(q(a[2]))})
    note = (f"{len(hp)} high-pass + {len(lp)} low-pass sections of butter({cfg.bandpass_order}) "
            f"{cfg.bandpass_hz[0]:g}-{cfg.bandpass_hz[1]:g} Hz, then the {cfg.notch_hz:g} Hz notch (Q {cfg.notch_q:g})")
    if cfg.filter_init == "first":
        # Starting the input history at the first sample x0 is the exact steady state for a constant
        # input only because the first section is a high-pass with numerator 1, -2, 1: its output,
        # and so every later section's state, is exactly 0. Check that this still holds.
        assert secs[0]["kind"] == "shift" and secs[0]["sign"] == -1, "first section must be high-pass"
    return FilterDesign(secs, sos, note, F, cfg.filter_init, cfg.state_frac)


def _section_step(s, p0, p1, p2, y1, y2, F, rnd):
    """One Direct Form I section, exactly as [hw] chip_pass. Works on Python ints or int64 arrays."""
    if s["kind"] == "shift":
        num = p0 + 2 * s["sign"] * p1 + p2
        acc = (num * s["g"]) if s["g"] is not None else (num << F)
    else:
        acc = s["b0"] * p0 + s["b1"] * p1 + s["b2"] * p2
    acc = acc - s["a1"] * y1 - s["a2"] * y2
    return (acc + rnd) >> F


def filter_wide(x16, design: FilterDesign):
    """Vectorised fixed-point filter: x16 (T x L int64, top 16 bits of the codes) -> the last
    section's output y (T x L int64), F fractional bits. Causal. At reset all section memory is 0;
    the input history is 0 (init "zero", spec 4) or the first sample (init "first", decided 31 Oct).
    The time loop is unavoidable for an IIR; all lanes are processed together."""
    x16 = np.asarray(x16, dtype=np.int64)
    T, L = x16.shape
    F = design.F
    rnd = np.int64(1) << (F - 1)
    z = lambda: np.zeros(L, np.int64)
    u1, u2 = z(), z()
    if design.init == "first" and T:
        u1, u2 = x16[0] << design.S, x16[0] << design.S
    ys = [[z(), z()] for _ in design.sections]
    out = np.empty((T, L), np.int64)
    for t in range(T):
        u0 = x16[t] << design.S
        p0, p1, p2 = u0, u1, u2
        for k, s in enumerate(design.sections):
            y1, y2 = ys[k]
            y0 = _section_step(s, p0, p1, p2, y1, y2, F, rnd)
            p0, p1, p2 = y0, y1, y2          # next section's input history = this section's outputs
            ys[k] = [y0, y1]
        u2, u1 = u1, u0
        out[t] = p0
    return out


def filter_out_int8(y, shift, S):
    """Output: shift right by (S + shift) with rounding, clip to INT8. S = the stored numbers'
    fractional bits (design.S).                                             [spec 4]"""
    sh = int(S) + int(shift)
    rnd = np.int64(1) << (sh - 1) if sh > 0 else np.int64(0)
    o = (np.asarray(y, np.int64) + rnd) >> np.int64(sh)
    return np.clip(o, INT8_MIN, INT8_MAX)


def choose_filter_shift(max_abs_y, S, shift_max=8):
    """Filter output shift from training data.                             [spec 4][hw calibrate]
    [hw] used a LEFT shift (gain) 0..7 and took the LARGEST one that keeps the loudest training
    output in range, i.e. the most resolution without clipping. The target shift is a RIGHT
    shift, so the same intent is the SMALLEST right shift that keeps the loudest training output
    inside -128..127 after rounding (OPEN-Q1)."""
    m = int(max_abs_y)
    for s in range(shift_max + 1):
        sh = S + s
        if ((m + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh) <= INT8_MAX:
            return s
    return shift_max


def filter_widths(design: FilterDesign, in_bits=16, n_imp=40000):
    """Worst-case widths so that nothing can overflow for ANY in_bits-bit input.        [spec 4]
    For section k: |y_k| <= 2^(in_bits-1) * 2^F * ||h_1..k||_1  +  sum_j 0.5 * ||e_j..k||_1
    where h_1..k is the response from the input to section k's output and e_j..k the response
    from section j's rounding error (injected inside j's recursion, so through 1/A_j) to section
    k's output, both with the quantized coefficients. The accumulator bound adds the size of
    every product. Returns a list of dicts and the bound check by simulation."""
    F = design.F
    xmax = 1 << (in_bits - 1)
    secs = design.sections

    def ba(s):
        if s["kind"] == "shift":
            g = (s["g"] if s["g"] is not None else (1 << F)) / (1 << F)
            b = np.array([1.0, 2.0 * s["sign"], 1.0]) * g
        else:
            b = np.array([s["b0"], s["b1"], s["b2"]]) / (1 << F)
        a = np.array([1.0, s["a1"] / (1 << F), s["a2"] / (1 << F)])
        return b, a

    imp = np.zeros(n_imp); imp[0] = 1.0
    h = imp.copy()
    hs = []
    for s in secs:
        b, a = ba(s)
        h = lfilter(b, a, h)
        hs.append(h.copy())
    S = design.S
    out, ymax_prev = [], float(xmax) * (1 << S)       # the input history u = x << S
    pin_bits = in_bits + S
    for k, s in enumerate(secs):
        L1 = np.abs(hs[k]).sum()
        err = 0.0
        for j in range(k + 1):
            bj, aj = ba(secs[j])
            e = lfilter([1.0], aj, imp)                      # rounding error through j's recursion
            for m in range(j + 1, k + 1):
                bm, am = ba(secs[m])
                e = lfilter(bm, am, e)
            err += 0.5 * np.abs(e).sum()
        ymax = xmax * (1 << S) * L1 + err
        state_bits = int(np.ceil(np.log2(ymax + 1))) + 1
        # accumulator: every term at its largest
        if s["kind"] == "shift":
            num = 4.0 * ymax_prev
            nterm = num * (abs(s["g"]) if s["g"] is not None else (1 << F))
            mults = 2 + (1 if s["g"] is not None else 0)
            mul_ops = (f"a1,a2: {16}x{state_bits}" +
                       (f"; g: 16x{int(np.ceil(np.log2(num + 1))) + 1}" if s["g"] is not None else ""))
        else:
            nterm = (abs(s["b0"]) + abs(s["b1"]) + abs(s["b2"])) * ymax_prev
            mults = 5
            mul_ops = f"b0,b1,b2: 16x{pin_bits}; a1,a2: 16x{state_bits}"
        acc = nterm + (abs(s["a1"]) + abs(s["a2"])) * ymax
        out.append(dict(section=k, kind=s["kind"], L1_gain=round(L1, 4),
                        state_max_bound=int(np.ceil(ymax)), state_bits=state_bits,
                        input_bits=pin_bits, acc_bits=int(np.ceil(np.log2(acc + 1))) + 1,
                        multiplies=mults, multiplier_operands=mul_ops))
        ymax_prev, pin_bits = ymax, state_bits
    # simulation check: the worst-case input for the last section, and random full scale
    worst = np.where(hs[-1][:4000][::-1] >= 0, xmax - 1, -xmax).astype(np.int64)
    rng = np.random.default_rng(0)
    rand = rng.integers(-xmax, xmax, size=4000)
    x = np.stack([np.r_[worst, worst], np.r_[rand, rand]], axis=1)
    y_obs = _all_section_outputs(x, design)
    for k, row in enumerate(out):
        row["state_max_seen"] = int(np.abs(y_obs[k]).max())
        assert row["state_max_seen"] <= row["state_max_bound"], "width bound broken"
    return out


def _all_section_outputs(x16, design):
    """Every section's output for an input block (used only by filter_widths)."""
    T, L = x16.shape
    F = design.F
    rnd = np.int64(1) << (F - 1)
    u1 = np.zeros(L, np.int64); u2 = u1.copy()
    ys = [[np.zeros(L, np.int64), np.zeros(L, np.int64)] for _ in design.sections]
    rec = [np.zeros((T, L), np.int64) for _ in design.sections]
    for t in range(T):
        u0 = x16[t].astype(np.int64) << design.S
        p0, p1, p2 = u0, u1, u2
        for k, s in enumerate(design.sections):
            y1, y2 = ys[k]
            y0 = _section_step(s, p0, p1, p2, y1, y2, F, rnd)
            rec[k][t] = y0
            p0, p1, p2 = y0, y1, y2
            ys[k] = [y0, y1]
        u2, u1 = u1, u0
    return rec


# =====================================================================================
# Per-sample features and banks                                           [spec 5][spec 6]
# =====================================================================================
def mitchell_sq(a, segments="one"):
    """|x|^2 approximation for a in 0..128.                                [spec 5][rtl][review D4]
    one: (a << (p+1)) - (1 << 2p), p = position of the leading 1; 0 -> 0.
    two: if the bit below the leading 1 is 1: m << (p+2), m = a - 2^p; else as one."""
    a = int(a)
    if a == 0:
        return 0
    p = a.bit_length() - 1
    if segments == "two" and p >= 1 and (a >> (p - 1)) & 1:
        return (a - (1 << p)) << (p + 2)
    return (a << (p + 1)) - (1 << (2 * p))


POW_LUT = {s: np.array([mitchell_sq(a, s) for a in range(129)], dtype=np.int64) for s in ("one", "two")}


def per_sample_features(x, x1, d0, segments="one"):
    """x, x1: INT8 sample and previous sample; d0: previous difference (int).   [spec 5][rtl]
    SABS = |x|; WL = |x - x1| (9-bit difference); ZC = MSB(x) xor MSB(x1), 0 counts as positive;
    SSC strict: count only strictly-up to strictly-down or the reverse, a zero difference
    clears both state bits; POW = Mitchell square. Arrays or ints. Returns (features..., d1)."""
    d1 = x - x1
    a = np.abs(x)
    ssc = ((d1 > 0) & (d0 < 0)) | ((d1 < 0) & (d0 > 0))
    return a, np.abs(d1), ((x < 0) != (x1 < 0)).astype(np.int64), ssc.astype(np.int64), \
        POW_LUT[segments][a], d1


def chip_features(x8, cfg: ChipConfig, codes=None):
    """Vectorised Stage 1 + window sums for whole recordings.             [spec 5][spec 6][spec 9]
    x8: (T, R, C) INT8 samples entering Stage 1 (after the filter or the legacy direct path).
    codes: (T, R, C) 24-bit codes, needed only for the code24 / code_low8 dead flag.
    Returns dict with
      banks (nb, R, C, 5)  bank sums, bank j = samples j*B .. j*B+B-1
      win   (nb, R, C, 5)  window sum at read-out r = banks r-N+1 .. r (banks before 0 = 0)
      rest_sum (nb, R)     sum over channels of the SABS window sums           [spec 8]
      dead  (nb, R, C)     dead-channel flag at each read-out                    [spec 9]
      valid (nb,)          read-out r is kept (r >= warm-up)                     [spec 10]
    History after reset: x1 = 0, d0 = 0                                          [spec 5][review D3]
    The history carries across bank boundaries: the step between banks is counted.  [spec 6]"""
    x = np.asarray(x8, dtype=np.int64)
    if x.ndim == 2:
        x = x[:, None, :]
    T, R, C = x.shape
    B, N = cfg.bank_len, cfg.n_banks
    nb = T // B
    x = x[: nb * B]
    zero = np.zeros((1, R, C), np.int64)
    x1 = np.concatenate([zero, x[:-1]])
    d1 = x - x1
    d0 = np.concatenate([zero, d1[:-1]])
    f = np.stack(per_sample_features(x, x1, d0, cfg.mitchell)[:5], axis=-1)     # (T, R, C, 5)
    banks = f.reshape(nb, B, R, C, N_TYPES).sum(axis=1)
    assert nb == 0 or np.all(banks.max(axis=(0, 1, 2)) < (1 << np.array(BANK_BITS))), \
        "bank field overflow [spec 6]"
    win = banks.copy()
    for j in range(1, N):
        win[j:] += banks[:-j]
    rest_sum = win[..., 0].sum(axis=2)
    if cfg.dead_rule == "wl_int8":                       # current RTL: WL window sum = 0   [rtl]
        dead = win[..., 1] == 0
    elif cfg.dead_rule in ("code24", "code_low8"):
        assert codes is not None, "the dead flag needs the 24-bit codes"
        c = np.asarray(codes, np.int64)
        if c.ndim == 2:
            c = c[:, None, :]
        c = c[: nb * B]
        if cfg.dead_rule == "code_low8":
            c = c & 0xFF
        ch = np.zeros_like(c, dtype=bool)
        ch[1:] = c[1:] != c[:-1]                         # first sample after reset: no change (OPEN-Q3)
        bch = ch.reshape(nb, B, R, C).any(axis=1)
        wch = bch.copy()
        for j in range(1, N):
            wch[j:] |= bch[:-j]
        dead = ~wch
    else:
        dead = np.zeros((nb, R, C), bool)
    valid = np.arange(nb) >= cfg.warmup
    return dict(banks=banks, win=win, rest_sum=rest_sum, dead=dead, valid=valid)


def window_matrix(win):
    """(nb, R, C, 5) window sums -> (nb, R, D) features, channel-major f = 5*ch + type.   [lgm]"""
    nb, R, C, _ = win.shape
    return win.reshape(nb, R, C * N_TYPES)


# =====================================================================================
# Requantizer                                                             [spec 7]
# =====================================================================================
def requant(Fsum, mu, k):
    """d = F - mu'; k > 0: (d + 2^(k-1)) >> k (arithmetic); k <= 0: d << -k; clip to INT8.
    In hardware the left shift saturates instead of wrapping; the clip of the exact value is
    the same number.                                                         [spec 7][review C4]"""
    d = np.asarray(Fsum, np.int64) - np.asarray(mu, np.int64)
    k = np.asarray(k, np.int64)
    kp = np.maximum(k, 1)
    pos = (d + (np.int64(1) << (kp - 1))) >> kp
    neg = d << np.maximum(-k, 0)
    return np.clip(np.where(k > 0, pos, neg), INT8_MIN, INT8_MAX)


def rule_max(Fcal, cfg):
    """Current rule of [lgm] calibrate_norm, with k allowed <= 0:        [spec 7]
    mu' = rounded mean; k = smallest shift with which no calibration window clips
    (ceil(log2(max|F - mu| / 127)), at least 1 in the legacy preset)."""
    mu = np.round(Fcal.mean(axis=0)).astype(np.int64)
    dev = np.abs(Fcal - mu).max(axis=0)
    k = np.ceil(np.log2(np.maximum(dev, 1) / INT8_MAX)).astype(np.int64)
    return mu, k


def rule_percentile(Fcal, cfg, pct=99.9):
    """Alternative rule ([hw] ShiftNormalizer): spread = 99.9th percentile of |F - mu|."""
    mu = np.round(Fcal.mean(axis=0)).astype(np.int64)
    dev = np.maximum(np.percentile(np.abs(Fcal - mu), pct, axis=0), 1.0)
    return mu, np.ceil(np.log2(dev / INT8_MAX)).astype(np.int64)


def calibrate_requant(Fcal, cfg: ChipConfig, rule=rule_max):
    """mu' and k from calibration windows (n x D). The rule is replaceable.   [spec 7]"""
    Fcal = np.asarray(Fcal, np.int64)
    mu, k = rule(Fcal, cfg)
    k = np.clip(k, cfg.k_min, cfg.k_max)
    assert mu.min() >= 0 and mu.max() < (1 << cfg.mu_bits), "mu' does not fit its register"
    return mu, k


# =====================================================================================
# Rest gate                                                               [spec 8][rest]
# =====================================================================================
def calibrate_t_rest(rest_sums, C, window_samples):
    """[rest]: threshold = mean of log(strength) over the user's rest recording, margin m = 0,
    strength = mean MAV over the channels. On the chip, strength = rest_sum / (C * window).
    rest  <=>  log(strength) < thr  <=>  rest_sum < C * window * exp(thr).
    T_REST = ceil(C * window * exp(thr)) makes the integer compare exactly equivalent.  [spec 8]
    rest_test adds 1e-12 inside the log; a rest window with rest_sum = 0 then pulls the
    threshold down hard (OPEN-Q4): such windows are counted and reported."""
    s = np.asarray(rest_sums, np.float64) / (C * window_samples)
    thr = np.mean(np.log(s + 1e-12))
    t = int(np.ceil(C * window_samples * np.exp(thr)))
    return t, int(np.sum(np.asarray(rest_sums) == 0))


# =====================================================================================
# Sample-by-sample reference: mirrors the RTL counters                    [lgm structure]
# =====================================================================================
class ChipRef:
    """One sample (C codes, or C INT8 samples in the legacy direct path) per step().
    Returns None, or a read-out dict every B samples. Python ints throughout."""

    def __init__(self, cfg: ChipConfig, C, mu, k, filt_shift=None, t_rest=None, W=None, b=None):
        self.cfg, self.C, self.D = cfg, C, N_TYPES * C
        self.mu = [int(v) for v in mu]
        self.k = [int(v) for v in k]
        self.t_rest = t_rest
        self.W, self.b = W, b
        if cfg.input_path == "filter":
            assert filt_shift is not None
            self.design = design_filter(cfg)
            self.shift = int(filt_shift)
            S = len(self.design.sections)
            self.u = [[0, 0] for _ in range(C)]                          # x16[n-1], x16[n-2] (<< F)
            self.first = [True] * C                                      # next sample is the first
            self.ys = [[[0, 0] for _ in range(S)] for _ in range(C)]     # per section y[n-1], y[n-2]
        self.x1 = [0] * C                                                # history RAM: x[n-1]
        self.d0p = [0] * C                                               # history RAM: d0 > 0
        self.d0n = [0] * C                                               #              d0 < 0
        self.prev_code = [None] * C                                      # dead check
        N = cfg.n_banks
        self.bank = [[[0] * N_TYPES for _ in range(C)] for _ in range(N)]
        self.bank_changed = [[False] * C for _ in range(N)]
        self.ptr = 0                                                     # 2-bit bank pointer
        self.cnt = 0                                                     # sample in the bank
        self.readouts = 0

    def _filter(self, ch, x16):
        F = self.design.F
        rnd = 1 << (F - 1)
        u0 = x16 << self.design.S
        if self.first[ch]:
            self.first[ch] = False
            if self.design.init == "first":                      # load the first sample into both
                self.u[ch] = [u0, u0]                            # input-history registers
        p0, p1, p2 = u0, self.u[ch][0], self.u[ch][1]
        for k, s in enumerate(self.design.sections):
            y1, y2 = self.ys[ch][k]
            if s["kind"] == "shift":
                num = p0 + 2 * s["sign"] * p1 + p2
                acc = num * s["g"] if s["g"] is not None else num << F
            else:
                acc = s["b0"] * p0 + s["b1"] * p1 + s["b2"] * p2
            acc = acc - s["a1"] * y1 - s["a2"] * y2
            y0 = (acc + rnd) >> F
            p0, p1, p2 = y0, y1, y2
            self.ys[ch][k] = [y0, y1]
        self.u[ch] = [u0, self.u[ch][0]]
        sh = self.design.S + self.shift
        o = (p0 + ((1 << (sh - 1)) if sh > 0 else 0)) >> sh
        return max(INT8_MIN, min(INT8_MAX, o))

    def step(self, codes):
        """codes: C ints (24-bit). Returns (x8 list, read-out or None)."""
        cfg = self.cfg
        if cfg.input_path == "filter":
            x8 = [self._filter(ch, int(codes[ch]) >> cfg.filt_in_shift) for ch in range(self.C)]
        else:
            x8 = [max(INT8_MIN, min(INT8_MAX, int(codes[ch]) >> min(cfg.int8_shift, 16))) for ch in range(self.C)]
        return x8, self.step_int8(x8, codes)

    def step_int8(self, x8, codes=None):
        cfg, C, B, N = self.cfg, self.C, self.cfg.bank_len, self.cfg.n_banks
        lut = POW_LUT[cfg.mitchell]
        for ch in range(C):                                      # Stage 1, one channel per cycle
            x = int(x8[ch])
            xp = self.x1[ch]
            d1 = x - xp
            ssc = int((d1 > 0 and self.d0n[ch]) or (d1 < 0 and self.d0p[ch]))
            f = (abs(x), abs(d1), int((x < 0) != (xp < 0)), ssc, int(lut[abs(x)]))
            row = self.bank[self.ptr][ch]
            for t in range(N_TYPES):
                row[t] = f[t] if self.cnt == 0 else row[t] + f[t]        # first sample overwrites
                assert row[t] < (1 << BANK_BITS[t]), "bank field overflow"
            self.x1[ch], self.d0p[ch], self.d0n[ch] = x, int(d1 > 0), int(d1 < 0)
            if cfg.dead_rule in ("code24", "code_low8"):
                c = int(codes[ch]) & (0xFF if cfg.dead_rule == "code_low8" else ~0)
                changed = self.prev_code[ch] is not None and c != self.prev_code[ch]
                self.bank_changed[self.ptr][ch] = changed if self.cnt == 0 else (self.bank_changed[self.ptr][ch] or changed)
                self.prev_code[ch] = c
        if self.cnt < B - 1:
            self.cnt += 1
            return None
        # ---- read-out: window = the last N banks (banks before reset count as 0)   [spec 6]
        self.cnt = 0
        r = self.readouts
        win = [[0] * N_TYPES for _ in range(C)]
        changed = [False] * C
        for j in range(min(N, r + 1)):
            bi = (self.ptr - j) % N
            for ch in range(C):
                for t in range(N_TYPES):
                    win[ch][t] += self.bank[bi][ch][t]
                changed[ch] = changed[ch] or self.bank_changed[bi][ch]
        Fsum = np.array([win[ch][t] for ch in range(C) for t in range(N_TYPES)], np.int64)
        q = np.array([_requant_scalar(int(Fsum[f]), self.mu[f], self.k[f]) for f in range(self.D)], np.int64)
        rest_sum = sum(win[ch][0] for ch in range(C))
        if cfg.dead_rule == "wl_int8":
            dead = [win[ch][1] == 0 for ch in range(C)]
        elif cfg.dead_rule in ("code24", "code_low8"):
            dead = [not changed[ch] for ch in range(C)]
        else:
            dead = [False] * C
        out = dict(index=r, valid=r >= cfg.warmup, F=Fsum, q=q, rest_sum=rest_sum,
                   rest=(rest_sum < self.t_rest) if (cfg.rest_gate and self.t_rest is not None) else False,
                   dead=dead, bank=[row[:] for row in self.bank[self.ptr]])
        if cfg.chip_classes:                                          # legacy Stage 3   [rtl]
            out.update(chip_lda(q, self.W, self.b, cfg.chip_classes))
        self.ptr = (self.ptr + 1) % N                                 # advance the bank pointer
        self.readouts += 1
        return out


def _requant_scalar(F, mu, k):
    d = F - mu
    v = (d + (1 << (k - 1))) >> k if k > 0 else d << (-k)
    return max(INT8_MIN, min(INT8_MAX, v))


def chip_lda(q, W, b, n_cls):
    """Legacy on-chip Stage 3: bias preload, D INT8 x INT8 multiply-adds into INT32,
    strict running argmax (ties go to the lower class).                       [rtl][lgm]"""
    scores, best, mx = [], 0, 0
    for c in range(n_cls):
        acc = int(b[c])
        for f in range(len(q)):
            acc += int(q[f]) * int(W[c, f])
            assert INT32_MIN <= acc <= INT32_MAX, "accumulator overflow"
        if c == 0 or acc > mx:
            mx, best = acc, c
        scores.append(acc)
    return dict(cls=best, scores=scores)


# =====================================================================================
# Whole-recording vectorised chip (for training) and the cross-check       [check A]
# =====================================================================================
def chip_run(codes, cfg: ChipConfig, filt_shift=None, design=None):
    """Vectorised chip on whole recordings. codes: (T, C) or (T, R, C) 24-bit codes.
    Returns x8 (T, R, C) and the chip_features dict."""
    c = np.asarray(codes, np.int64)
    if c.ndim == 2:
        c = c[:, None, :]
    T, R, C = c.shape
    if cfg.input_path == "filter":
        design = design or design_filter(cfg)
        y = filter_wide(code_to_filter_input(c, cfg).reshape(T, R * C), design)
        x8 = filter_out_int8(y, filt_shift, design.S).reshape(T, R, C)
    else:
        x8 = ads_code_to_int8(c, cfg.int8_shift)
    return x8, chip_features(x8, cfg, c)


def selfcheck(recordings, cfg: ChipConfig, filt_shift=None, mu=None, k=None, t_rest=None,
              W=None, b=None, verbose=True):
    """Check A: the sample-by-sample reference and the vectorised version give identical
    filtered samples, bank sums, window sums, INT8 features, rest flags and dead flags.
    recordings: list of (T, C) code arrays. Asserts; returns the number of read-outs compared."""
    n = 0
    for rec in recordings:
        rec = np.asarray(rec, np.int64)
        T, C = rec.shape
        D = N_TYPES * C
        x8v, fv = chip_run(rec, cfg, filt_shift)
        Fv = window_matrix(fv["win"])[:, 0, :]
        mu_ = np.zeros(D, np.int64) if mu is None else mu
        k_ = np.full(D, max(cfg.k_min, 1) if cfg.k_min > 0 else 0, np.int64) if k is None else k
        qv = requant(Fv, mu_, k_)
        tr = t_rest if t_rest is not None else int(np.median(fv["rest_sum"][:, 0])) + 1
        ref = ChipRef(cfg, C, mu_, k_, filt_shift, tr, W, b)
        r = 0
        for t in range(T):
            x8, out = ref.step(rec[t])
            assert list(x8v[t, 0]) == x8, f"filtered sample mismatch at t={t}"
            if out is None:
                continue
            assert out["bank"] == fv["banks"][r, 0].tolist(), f"bank sums mismatch at read-out {r}"
            assert np.array_equal(out["F"], Fv[r]), f"window sums mismatch at read-out {r}"
            assert np.array_equal(out["q"], qv[r]), f"INT8 features mismatch at read-out {r}"
            assert out["rest_sum"] == fv["rest_sum"][r, 0]
            if cfg.rest_gate:
                assert out["rest"] == (fv["rest_sum"][r, 0] < tr)
            assert out["dead"] == fv["dead"][r, 0].tolist(), f"dead flags mismatch at read-out {r}"
            assert out["valid"] == fv["valid"][r]
            r += 1
        assert r == fv["banks"].shape[0]
        n += r
    if verbose:
        print(f"  check A ({cfg.name}): reference == vectorised on {len(recordings)} recording(s), "
              f"{n} read-outs: filtered samples, bank sums, window sums, INT8 features, rest, dead flags")
    return n


# =====================================================================================
# PROCESSOR MODEL (classifier route (a))                                 [spec 11][win][rest]
# =====================================================================================
LABELS = ["Hand Close", "Hand Open", "Index", "Index Little", "Peace", "Radial Deviation", "Right Angle",
          "Thumb Little", "Thumb Up", "Ulnar Deviation", "Wrist Extension", "Wrist Flexion", "Rest"]
GROUPS = [("G1 grip", ["Hand Close", "Thumb Up", "Index", "Index Little", "Peace", "Right Angle"]),
          ("G2 opening", ["Hand Open", "Wrist Extension"]), ("G3 radial", ["Radial Deviation"]),
          ("G4 thumb-little", ["Thumb Little"]), ("G5 ulnar", ["Ulnar Deviation"]),
          ("G6 flexion", ["Wrist Flexion"]), ("Rest", ["Rest"])]                       # [win][rest]
K_ALL = len(LABELS)                                    # 13 = 12 gestures + Rest           [rest]
REST = LABELS.index("Rest")
GROUP_OF = np.zeros(K_ALL, int)
for _g, (_n, _m) in enumerate(GROUPS):
    for _x in _m:
        GROUP_OF[LABELS.index(_x)] = _g
REST_GROUP = GROUP_OF[REST]
ANCHOR = [LABELS.index(a) for a in ("Hand Open", "Wrist Flexion", "Wrist Extension")]    # [win]
CONFIRM_M = 4                                          # confirm-4 at 50 ms steps          [spec 11]


def class_stats(X, y, K=K_ALL):
    """Class summaries n, sum, sum of outer products.                     [win]"""
    D = X.shape[1]
    n, s, q = np.zeros(K), np.zeros((K, D)), np.zeros((K, D, D))
    for c in np.unique(y):
        Xc = X[y == c]
        n[c], s[c], q[c] = len(Xc), Xc.sum(0), Xc.T @ Xc
    return n, s, q


def unit(st):
    t = st[0].sum()
    return st[0] / t, st[1] / t, st[2] / t


def blend(G, C, share):
    """Method C: (1 - share) x general + share x calibration, each normalised.   [win model_for]"""
    a, b = (1 - share) / G[0].sum(), share / C[0].sum()
    return tuple(a * G[i] + b * C[i] for i in range(3))


def lda(n, s, q):
    """LDA from class summaries (pooled covariance + tiny ridge).          [win]"""
    cls = np.flatnonzero(n > 0)
    mu = s[cls] / n[cls, None]
    sigma = sum(q[c] - np.outer(s[c], s[c]) / n[c] for c in cls) / n[cls].sum()
    D = sigma.shape[0]
    sigma = sigma + np.eye(D) * 1e-6 * np.trace(sigma) / D
    W = np.linalg.solve(sigma, mu.T).T
    return W, -0.5 * np.einsum("kd,kd->k", mu, W), cls


def predict_float(model, Z):
    W, b, cls = model
    return cls[np.argmax(Z @ W.T + b[None, :], 1)]


def quantize_rows(model, sx=1.0):
    """INT8 weights with one power-of-2 scale per class row.               [win predict_rows]"""
    W, b, cls = model
    e = np.ceil(np.log2(np.abs(W).max(1) / 127 + 1e-30)).astype(int)
    Wq = np.clip(np.round(W / 2.0 ** e[:, None]), -128, 127).astype(np.int64)
    bq = np.round(b / (2.0 ** e * sx)).astype(np.int64)
    return Wq, bq, e, cls


def predict_rows(model, Z, sx):
    """[win] predict_rows, unchanged: scores per class row, rescaled by 2^(e - e_min)."""
    Wq, bq, e, cls = quantize_rows(model, sx)
    xq = np.clip(np.round(Z / sx), -128, 127).astype(np.int64)
    return cls[np.argmax((xq @ Wq.T + bq[None, :]) * (2 ** (e - e.min()))[None, :], 1)]


def processor_predict(model_q, xq, rest_flag=None):
    """The processor on the chip's INT8 features (sx = 1, already INT8).    [spec 11]
    model_q = quantize_rows(model, 1.0). The rest flag overrides to Rest."""
    Wq, bq, e, cls = model_q
    s = (np.asarray(xq, np.int64) @ Wq.T + bq[None, :]) * (2 ** (e - e.min()))[None, :]
    p = cls[np.argmax(s, 1)]
    if rest_flag is not None:
        p = np.where(np.asarray(rest_flag, bool), REST, p)
    return p


def confirm(g, M=CONFIRM_M):
    """confirm-M on one recording; before the first confirmed decision the arm does nothing (-1).  [win]"""
    out, state, run, last = np.empty_like(g), -1, 0, None
    for i, x in enumerate(g):
        run = run + 1 if x == last else 1
        last = x
        if x != state and run >= M:
            state = x
        out[i] = state
    return out


def anchor_scaler(X, y):
    """[win] per-person re-centring on the anchor gestures (float pipeline rows only)."""
    m = np.isin(y, ANCHOR)
    return X[m].mean(0), X[m].std(0) + 1e-9


# =====================================================================================
# Self test (synthetic data; FORS-EMG runs live in fors_chip_study.py)
# =====================================================================================
def synthetic_volts(T, C, rng, level_v=200e-6, dc_v=None):
    """EMG-like test signal: band-limited noise with random channel gains (volts)."""
    from scipy.signal import butter as _b, sosfilt as _s
    sos = _b(4, [30, 300], btype="bandpass", fs=1000, output="sos")
    x = _s(sos, rng.standard_normal((T, C)), axis=0) * level_v * rng.uniform(0.3, 3, C)[None, :]
    if dc_v is not None:
        x = x + dc_v[None, :]
    return x


def main():
    np.set_printoptions(linewidth=140)
    for preset in ("target", "legacy"):
        cfg = make_config(preset)
        print(f"\n=== preset {preset} ===")
        print(cfg.to_json())
    cfg = make_config("target")
    des = design_filter(cfg)
    print(f"\nFilter: {des.order_note}")
    for s in des.sections:
        print("  ", s)
    print("\nFilter widths for any 16-bit input (bound; simulation max must stay below):")
    for row in filter_widths(des):
        print("  ", row)
    rng = np.random.default_rng(1)
    recs = []
    for i in range(3):
        v = synthetic_volts(3000, 8, rng, dc_v=rng.uniform(-0.05, 0.05, 8))
        recs.append(volts_to_codes(v, cfg))
    recs[1][:, 3] = 0                                     # a missing channel (zeros)
    recs[2][:, 5] = CODE_MAX                              # a saturated channel
    _, f = chip_run(recs[0], cfg, 6)
    mu, k = calibrate_requant(window_matrix(f["win"])[f["valid"], 0], cfg)
    selfcheck(recs, cfg, filt_shift=6, mu=mu, k=k)
    leg = make_config("legacy")
    selfcheck([r >> 0 for r in recs], leg, mu=np.zeros(40, np.int64), k=np.ones(40, np.int64),
              W=np.ones((6, 40), np.int64), b=np.zeros(6, np.int64))


if __name__ == "__main__":
    main()
