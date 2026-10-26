"""
FORS-EMG — bit-exact chip LDA vs the validated INT8 LDA, subject-independent LOSO
=================================================================================
Data loading, causal filter, ADC full scale, gesture labels and the LDA estimator
come from K_4_LDA_Analysis.py (revision 3), exactly as in K_4_CNN_LOSO.py.
Everything the chip does comes from lda_golden_model.py:
  * each recording is streamed sample by sample (no per-window restart), so
    WL / ZC / SSC use the true previous sample, as the hardware does;
  * 196-sample window, 98-sample stride (ping-pong banks), first half-window dropped;
  * shift normaliser (mu_f, k_f) and INT8 LDA with one weight scale, 6-class argmax.

Split (same as K_4_CNN_LOSO.py): trials 1-4 of the 18 training subjects fit the
normaliser and the LDA; every trial of the held-out subject tests.
The validated pipeline (its own windows and features) runs on the same split, so
the two are compared subject by subject.
Prints results only; writes no files.

Run:  python K_4_LDA_Golden_LOSO.py
      (K_4_LDA_Analysis.py and lda_golden_model.py in the same folder)
"""

import time
import numpy as np
from scipy.stats import t as student_t, wilcoxon
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from tqdm import tqdm

import K_4_LDA_Analysis as B
import lda_golden_model as G

TRAIN_TRIALS = (1, 2, 3, 4)
LDA_HW_TYPES = ["SABS", "WL", "ZC_NODB", "SSC", "POW_M1"]
C = B.N_CHANNELS
D = G.N_TYPES * C


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_recordings():
    """Filtered recordings (samples x channels) and the ADC full scale, as in K_4_CNN_LOSO.build_windows."""
    subjects = B.load_all_subjects(B.BASE_DIR)
    filt = B._design_filters()
    recs, full_scale = [], 0.0
    for s in tqdm(subjects, desc="Filtering"):
        for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
            sig = B.apply_preprocessing_filter(rec.T, filt, "causal").T
            full_scale = max(full_scale, float(np.abs(sig).max()))
            recs.append((sig, lbl, s["subject_id"], m["trial"]))
    return recs, [s["subject_id"] for s in subjects], full_scale


def adc(sig, full_scale):
    """B.quantize_signal on the whole recording, laid out (1, channels, samples) like a window batch -> int8 codes (8-bit ADC, not SIGNAL_BITS)."""
    q = np.asarray(B.quantize_signal(sig.T[None].astype(np.float32), full_scale, bits=8))[0].T
    assert np.array_equal(q, np.round(q)), "quantize_signal must return integer ADC codes"
    assert q.min() >= -128 and q.max() <= 127
    return q.astype(np.int8)


def build_features(recs, full_scale):
    """Per window: chip features (golden model) and the validated pipeline's features."""
    chip = dict(raw=[], y=[], sid=[], trial=[])
    ref = dict(X=[], y=[], sid=[], trial=[])
    probe, ref_len = None, None
    for sig, lbl, sid, trial in tqdm(recs, desc="Features"):
        w, yy = B.windows_from_recordings([sig], [lbl])       # also maps / drops gestures as the LDA script does
        yy = np.asarray(yy)
        if len(yy) == 0:
            continue
        assert len(np.unique(yy)) == 1, "one gesture per recording expected"
        w = np.asarray(w, np.float32)
        ref_len = w.shape[-1]
        ref["X"].append(B.extract_hw_features(B.quantize_signal(w, full_scale, bits=8), LDA_HW_TYPES))
        ref["y"].append(yy); ref["sid"].append(np.full(len(yy), sid)); ref["trial"].append(np.full(len(yy), trial))

        q8 = adc(sig, full_scale)
        raw, full = G.window_features(q8, C)
        n = int(full.sum())
        chip["raw"].append(raw[full]); chip["y"].append(np.full(n, yy[0]))
        chip["sid"].append(np.full(n, sid)); chip["trial"].append(np.full(n, trial))
        if probe is None:
            probe = (q8, sid)
    cat = lambda d: {k: np.concatenate(v) if k != "X" and k != "raw" else np.vstack(v) for k, v in d.items()}
    return cat(chip), cat(ref), probe, ref_len


# ─── Models ───────────────────────────────────────────────────────────────────

def linear_form(lda):
    """Float (n_cls x D, n_cls) linear form from B.lda_as_linear_layer, whatever its orientation."""
    W, b = (np.asarray(a, dtype=float) for a in B.lda_as_linear_layer(lda))
    b = b.ravel()
    W = W if W.shape == (len(b), D) else W.T
    assert W.shape == (len(b), D)
    return W, b


def chip_fold(chip, t):
    tr = (chip["sid"] != t) & np.isin(chip["trial"], TRAIN_TRIALS)
    te = chip["sid"] == t
    mu, k = G.calibrate_norm(chip["raw"][tr])
    q_tr = G.normalize(chip["raw"][tr], mu, k)
    q_te = G.normalize(chip["raw"][te], mu, k)
    lda = B.build_lda().fit(q_tr, chip["y"][tr])
    Wf, bf = linear_form(lda)
    agree = np.mean(lda.classes_[np.argmax(q_tr @ Wf.T + bf, axis=1)] == lda.predict(q_tr))
    assert agree > 0.999, f"linear form does not reproduce lda.predict ({agree:.4f})"
    W, b = G.quantize_lda(Wf, bf)
    scores = q_te @ W.T + b
    assert scores.min() >= G.INT32_MIN and scores.max() <= G.INT32_MAX
    pred = lda.classes_[np.argmax(scores, axis=1)]           # first max = strict '>' in hardware
    n = len(lda.classes_)                                    # padded classes sit at INT32_MIN by design
    return accuracy_score(chip["y"][te], pred), dict(mu=mu, k=k, W=W, b=b, classes=lda.classes_,
                                                      max_score=int(np.abs(scores[:, :n]).max()))


def ref_fold(ref, t):
    """lda_reference from K_4_CNN_LOSO.py on the validated pipeline's windows."""
    tr = (ref["sid"] != t) & np.isin(ref["trial"], TRAIN_TRIALS)
    te = ref["sid"] == t
    sc = B.ShiftNormalizer()
    Xtr, Xte = sc.fit_transform(ref["X"][tr]), sc.transform(ref["X"][te])
    lda = B.build_lda().fit(Xtr, ref["y"][tr])
    q = B.calibrate_ptq([B.lda_as_linear_layer(lda)], Xtr, input_scale=1.0)
    return accuracy_score(ref["y"][te], lda.classes_[B.int8_infer(q, Xte)])


# ─── Statistics (as in K_4_CNN_LOSO.py) ───────────────────────────────────────

def mean_ci(v):
    v = np.asarray(v, dtype=float); n = len(v)
    return v.mean(), student_t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


def paired_row(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    m, h = mean_ci(d)
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = m - h, m + h
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"
    return [f"{m:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}", f"{(d > 0).sum()}/{(d < 0).sum()}", verdict]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 76)
    print("  FORS-EMG — bit-exact chip LDA vs validated INT8 LDA (paired LOSO)")
    print("=" * 76)
    recs, ids, full_scale = load_recordings()
    chip, ref, probe, ref_len = build_features(recs, full_scale)
    del recs
    print(f"\n  ADC full scale {full_scale:.4g}   channels {C}   features {D}")
    print(f"  Chip windows     : {len(chip['y']):,}  ({2 * G.SUB_WIN} samples, stride {G.SUB_WIN}, streamed)")
    print(f"  Validated windows: {len(ref['y']):,}  ({ref_len} samples, from windows_from_recordings)\n")

    acc_chip, acc_ref, hw = [], [], []
    for i, t in enumerate(ids):
        a_c, p = chip_fold(chip, t)
        a_r = ref_fold(ref, t)
        acc_chip.append(a_c); acc_ref.append(a_r); hw.append(p)
        print(f"  [{i+1:2d}/{len(ids)}] S{t:02d}  chip {a_c:.2%}  |  validated {a_r:.2%}", flush=True)

    print(f"\n{'─'*76}\n  Accuracy (mean over {len(ids)} held-out subjects; ± = 95% CI)\n{'─'*76}")
    rows = []
    for name, v in (("Chip LDA (golden model)", acc_chip), ("Validated INT8 LDA", acc_ref)):
        m, h = mean_ci(v)
        rows.append([name, f"{m*100:.2f} ± {h*100:.2f}"])
    print(tabulate(rows, headers=["Model", "Test %"], tablefmt="simple"))
    print()
    print(tabulate([["chip vs validated"] + paired_row(acc_chip, acc_ref)],
                   headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "Verdict"],
                   tablefmt="simple"))

    print(f"\n{'─'*76}\n  Hardware ranges over all folds\n{'─'*76}")
    mu_max = max(int(p["mu"].max()) for p in hw)
    print(f"  mu_f max {mu_max} ({mu_max.bit_length()} of 22 bits)   "
          f"k_f {min(int(p['k'].min()) for p in hw)}..{max(int(p['k'].max()) for p in hw)} (5 bits)")
    print(f"  max |score| {max(p['max_score'] for p in hw):,} (INT32 limit {G.INT32_MAX:,})   "
          f"|bias| max {max(int(np.abs(p['b'][:len(p['classes'])]).max()) for p in hw):,}")

    # cycle-faithful reference on one real recording, with the parameters of that subject's fold
    q8, sid = probe
    p = hw[ids.index(sid)]
    core = G.LdaCoreRef(C, p["mu"], p["k"], p["W"], p["b"])
    decisions = [d for s in q8 if (d := core.step(s)) is not None]
    raw, _ = G.window_features(q8, C)
    q_fast = G.normalize(raw, p["mu"], p["k"])
    for j, d in enumerate(decisions):
        assert np.array_equal(d["q"], q_fast[j])
        assert d["scores"] == [int(v) for v in q_fast[j] @ p["W"].T + p["b"]]
    print(f"\n  Reference core == vectorized model on a real S{sid:02d} recording "
          f"({len(decisions)} decisions): OK")
    if len(decisions) > 1:
        d = decisions[1]
        print(f"  Test vector (first full window): class {p['classes'][d['cls']]} (hw index {d['cls']}), "
              f"scores {d['scores']}")
        print(f"  x_q = {d['q'].tolist()}")

    print(f"\n  Per-subject test accuracy (%):")
    print(tabulate([[f"S{t:02d}", f"{c*100:.2f}", f"{r*100:.2f}"] for t, c, r in zip(ids, acc_chip, acc_ref)],
                   headers=["Subject", "Chip", "Validated"], tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time()-t0)/60:.1f} min")
    print("=" * 76)


if __name__ == "__main__":
    main()
