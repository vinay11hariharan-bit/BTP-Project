"""
FORS-EMG — hyperdimensional computing (HDC), Way 1 vs Way 2, subject-independent LOSO
====================================================================================
Way 1  HDC on the chip's features: the 40 INT8 values x_q (5 per channel) from the
       bit-exact Stage 1 + 2 model (lda_golden_model). One window -> one hypervector:
           H = majority_f( ID_f xor L[level(x_q,f)] )
Way 2  HDC on the samples, no hand-made features. Per sample, each channel's envelope
       (sum of |x| over the last ENV_LEN samples, INT8 ADC codes, >> ENV_SHIFT) -> level;
           S_t = majority_c( ID_c xor L[level_c(t)] )          (pattern across the arm)
           G_t = rot^2(S_{t-2}) xor rot(S_{t-1}) xor S_t       (3-step sequence)
           H   = majority over the 196 samples of the window of G_t
Both use the same windows (196 samples, stride 98, first half-window dropped), the
same split and the same classifier:
    train  trials 1-4 of the 18 training subjects
    val    trial 5 of the training subjects (picks the number of retraining passes)
    test   every trial of the held-out subject
    prototype_k = sign(sum of +-1 hypervectors of class k), then perceptron retraining;
    class = nearest prototype in Hamming distance.
The INT8 LDA on the same x_q (Way 1's input) is printed as the reference.
Prints results only; writes no files.

Run:  python K_4_HDC_Way1_vs_Way2.py
      (K_4_LDA_Analysis.py and lda_golden_model.py in the same folder)
"""

import os
import time
import numpy as np
from scipy.stats import t as student_t, wilcoxon
from tabulate import tabulate
from tqdm import tqdm

import K_4_LDA_Analysis as B
import lda_golden_model as G

# ─── Settings ─────────────────────────────────────────────────────────────────
DIM        = 2048          # hypervector length d
Q          = 16            # levels per value
ENV_LEN    = 64            # Way 2: envelope window, samples
ENV_SHIFT  = 5             # Way 2: level = min(Q-1, (sum of |x| over ENV_LEN samples) >> ENV_SHIFT)
NGRAM      = 3             # Way 2: samples per sequence
EPOCHS     = 30            # max retraining passes (best on validation is kept)
SEED       = 0
TRAIN_TRIALS = (1, 2, 3, 4)
VAL_TRIAL    = 5

C = B.N_CHANNELS
D = G.N_TYPES * C
K = B.N_CLASSES
STRIDE = G.SUB_WIN
DATA_DIR = B.BASE_DIR
if not os.path.isdir(DATA_DIR):            # fall back to the copy inside the repo
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "archive", "FORS-EMG Dataset", "FORS-EMG Dataset", "FORS-EMG")


# ─── Random hypervectors ──────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)


def level_vectors(q, d):
    """L_0 random; bit i flips once the level reaches its random threshold r_i in 1..2(q-1),
    so delta(L_a, L_b) = |a - b| / (2(q-1)) and the two ends are orthogonal."""
    L0 = rng.integers(0, 2, d, dtype=np.uint8)
    r = rng.integers(1, 2 * (q - 1) + 1, d)
    return np.stack([L0 ^ (r <= l).astype(np.uint8) for l in range(q)])


ID_F = rng.integers(0, 2, (D, DIM), dtype=np.uint8)        # Way 1: one tag per feature
ID_C = rng.integers(0, 2, (C, DIM), dtype=np.uint8)        # Way 2: one tag per channel
LV1 = level_vectors(Q, DIM)
LV2 = level_vectors(Q, DIM)
TIE = rng.integers(0, 2, DIM, dtype=np.uint8)              # fixed tie-break for even votes


def majority(count, n):
    """count = number of 1s out of n votes, per bit."""
    out = (2 * count > n).astype(np.uint8)
    tie = 2 * count == n
    out[tie] = np.broadcast_to(TIE, out.shape)[tie]
    return out


# ─── Data ─────────────────────────────────────────────────────────────────────
def load():
    subjects = B.load_all_subjects(DATA_DIR)
    filt = B._design_filters()
    recs, full_scale = [], 0.0
    for s in tqdm(subjects, desc="Filtering"):
        for rec, lbl, m in zip(s["recordings"], s["labels"], s["meta"]):
            sig = B.apply_preprocessing_filter(rec.T, filt, "causal").T
            full_scale = max(full_scale, float(np.abs(sig).max()))
            recs.append((sig, lbl, s["subject_id"], m["trial"]))
    out = []
    for sig, lbl, sid, trial in recs:
        q8 = np.asarray(B.quantize_signal(sig, full_scale, bits=8)).astype(np.int64)
        out.append((q8, lbl, sid, trial))
    return out, [s["subject_id"] for s in subjects]


def way2_encode_recording(q8):
    """Hypervector of every full window of one recording (Way 2). Returns (n_windows, DIM) uint8."""
    n = len(q8)
    a = np.abs(q8)
    cs = np.vstack([np.zeros((1, C), np.int64), np.cumsum(a, axis=0)])
    lo = np.maximum(np.arange(n) - ENV_LEN + 1, 0)
    env = cs[np.arange(n) + 1] - cs[lo]                                     # causal sum of |x|, ENV_LEN samples
    lev = np.minimum(env >> ENV_SHIFT, Q - 1)                               # (n, C)

    # S_t depends only on the C levels: encode each distinct level pattern once
    pat, s_id = np.unique(lev, axis=0, return_inverse=True)
    s_id = s_id.ravel()
    cnt = np.zeros((len(pat), DIM), np.int16)
    for c in range(C):
        cnt += ID_C[c] ^ LV2[pat[:, c]]
    S = majority(cnt, C)                                                    # (n_pat, DIM)

    # G_t depends only on (S_{t-2}, S_{t-1}, S_t): encode each distinct triple once
    prev = [np.concatenate([np.full(j, s_id[0]), s_id[:n - j]]) for j in range(NGRAM)]
    tri, g_id = np.unique(np.stack(prev, axis=1), axis=0, return_inverse=True)
    g_id = g_id.ravel()
    Gv = np.zeros((len(tri), DIM), np.uint8)
    for j in range(NGRAM):
        Gv ^= np.roll(S[tri[:, j]], j, axis=1)

    # window = two stride blocks of 98 samples, majority over 196 samples
    nb = n // STRIDE
    hist = np.zeros((nb, len(tri)), np.float32)
    np.add.at(hist, (np.arange(nb * STRIDE) // STRIDE, g_id[:nb * STRIDE]), 1)
    blk = hist @ Gv.astype(np.float32)                                      # ones per bit, per block
    win = blk[1:] + blk[:-1]
    return majority(win.astype(np.int32), 2 * STRIDE)


def build(recs):
    """Per window: Way 1 raw sums (for x_q), Way 2 hypervector, label, subject, trial."""
    raw, H2, y, sid, trial = [], [], [], [], []
    for q8, lbl, s, t in tqdm(recs, desc="Features + Way 2 encoding"):
        r, full = G.window_features(q8, C)
        h = way2_encode_recording(q8)
        r = r[full]
        assert len(r) == len(h)
        raw.append(r); H2.append(h)
        n = len(r)
        y.append(np.full(n, lbl)); sid.append(np.full(n, s)); trial.append(np.full(n, t))
    return (np.vstack(raw), np.vstack(H2), np.concatenate(y), np.concatenate(sid), np.concatenate(trial))


def way1_encode(xq):
    """(n, D) INT8 features -> (n, DIM) hypervectors."""
    lev = (xq + 128) >> 4 if Q == 16 else np.minimum(((xq + 128) * Q) >> 8, Q - 1)
    out = np.empty((len(xq), DIM), np.uint8)
    for i in range(0, len(xq), 4096):
        cnt = np.zeros((len(xq[i:i + 4096]), DIM), np.int16)
        for f in range(D):
            cnt += ID_F[f] ^ LV1[lev[i:i + 4096, f]]
        out[i:i + 4096] = majority(cnt, D)
    return out


# ─── Classifier ───────────────────────────────────────────────────────────────
def pm(h):
    return h.astype(np.float32) * 2 - 1


def predict(A, Hpm):
    """nearest prototype in Hamming distance; prototype = sign of the accumulator (0 -> tie vector)."""
    P = np.where(A > 0, 1.0, np.where(A < 0, -1.0, TIE * 2.0 - 1)).astype(np.float32)
    return np.argmax(Hpm @ P.T, axis=1)          # max agreement = min Hamming, first index on ties


def train(Htr, ytr, Hva, yva):
    Xtr, Xva = pm(Htr), pm(Hva)
    A = np.zeros((K, DIM), np.float32)
    for k in range(K):
        A[k] = Xtr[ytr == k].sum(axis=0)
    best = (np.mean(predict(A, Xva) == yva), 0, A.copy())
    for ep in range(1, EPOCHS + 1):
        p = predict(A, Xtr)
        wrong = p != ytr
        if not wrong.any():
            break
        for k in range(K):
            A[k] += Xtr[wrong & (ytr == k)].sum(axis=0)
            A[k] -= Xtr[wrong & (p == k)].sum(axis=0)
        acc = np.mean(predict(A, Xva) == yva)
        if acc > best[0]:
            best = (acc, ep, A.copy())
    return best[2], best[1]


# ─── Statistics ───────────────────────────────────────────────────────────────
def mean_ci(v):
    v = np.asarray(v, float)
    return v.mean(), student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def paired(a, b):
    d = (np.asarray(a) - np.asarray(b)) * 100
    m, h = mean_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    verdict = "better" if m - h > 0 else "worse" if m + h < 0 else "no detectable difference"
    return [f"{m:+.2f}", f"[{m - h:+.2f}, {m + h:+.2f}]", f"{p:.3f}", f"{(d > 0).sum()}/{(d < 0).sum()}", verdict]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"  FORS-EMG — HDC Way 1 (features) vs Way 2 (samples), LOSO   d = {DIM}, Q = {Q}")
    print("=" * 78)
    recs, ids = load()
    raw, H2, y, sid, trial = build(recs)
    del recs
    print(f"\n  Windows: {len(y):,}   classes: {K}   subjects: {len(ids)}")
    print(f"  Way 2 envelope: sum of |x| over {ENV_LEN} samples, level = min({Q - 1}, sum >> {ENV_SHIFT}), n-gram {NGRAM}\n")

    res = {"lda": [], "w1": [], "w2": []}
    eps = {"w1": [], "w2": []}
    for i, s in enumerate(ids):
        tr = (sid != s) & np.isin(trial, TRAIN_TRIALS)
        va = (sid != s) & (trial == VAL_TRIAL)
        te = sid == s

        # Way 1 input = chip x_q, normaliser fitted on training subjects only
        mu, k = G.calibrate_norm(raw[tr | va])
        xq = G.normalize(raw, mu, k)

        lda, W, b = G.fit_int8_lda(xq[tr | va], y[tr | va])
        res["lda"].append(np.mean(lda.classes_[np.argmax(xq[te] @ W.T + b, axis=1)] == y[te]))

        H1 = way1_encode(xq)
        A, e = train(H1[tr], y[tr], H1[va], y[va])
        res["w1"].append(np.mean(predict(A, pm(H1[te])) == y[te])); eps["w1"].append(e)

        A, e = train(H2[tr], y[tr], H2[va], y[va])
        res["w2"].append(np.mean(predict(A, pm(H2[te])) == y[te])); eps["w2"].append(e)

        print(f"  [{i + 1:2d}/{len(ids)}] S{s:02d}   LDA {res['lda'][-1]:.2%}   "
              f"HDC Way 1 {res['w1'][-1]:.2%}   HDC Way 2 {res['w2'][-1]:.2%}", flush=True)

    print(f"\n{'─' * 78}\n  Test accuracy (mean over {len(ids)} held-out subjects; ± = 95% CI)\n{'─' * 78}")
    rows = []
    for key, name in (("lda", "INT8 LDA on x_q (reference)"), ("w1", "HDC Way 1 (features)"),
                      ("w2", "HDC Way 2 (samples)")):
        m, h = mean_ci(res[key])
        e = f"{np.mean(eps[key]):.1f}" if key in eps else ""
        rows.append([name, f"{m * 100:.2f} ± {h * 100:.2f}", e])
    print(tabulate(rows, headers=["Model", "Test %", "Retrain passes"], tablefmt="simple"))
    print()
    print(tabulate([["Way 1 vs Way 2"] + paired(res["w1"], res["w2"]),
                    ["Way 1 vs LDA"] + paired(res["w1"], res["lda"]),
                    ["Way 2 vs LDA"] + paired(res["w2"], res["lda"])],
                   headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "Verdict"],
                   tablefmt="simple"))
    print(f"\n  Per-subject test accuracy (%):")
    print(tabulate([[f"S{s:02d}", f"{a * 100:.2f}", f"{b_ * 100:.2f}", f"{c * 100:.2f}"]
                    for s, a, b_, c in zip(ids, res["lda"], res["w1"], res["w2"])],
                   headers=["Subject", "LDA", "Way 1", "Way 2"], tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time() - t0) / 60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
