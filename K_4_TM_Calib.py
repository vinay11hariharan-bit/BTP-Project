"""
FORS-EMG — Tsetlin machine (TM) on the chip's features, with per-user calibration, LOSO
======================================================================================
Input: the 40 INT8 features x_q per window from the bit-exact Stage 1 + 2 model
(lda_golden_model), same windows and split as the LDA and HDC scripts.

Booleanisation (thermometer code): each feature f gets BITS thresholds t_f1 < ... at
evenly spaced quantiles of the training subjects' x_q; bit j = [x_q,f >= t_fj].
Literals = the bits and their negations: L = 2 * 40 * BITS.

Tsetlin machine (multi-class, standard rules):
  * per class CLAUSES clauses, first half positive, second half negative polarity
  * every (clause, literal) has a Tsetlin automaton: a counter 1..2N, include if > N
  * clause = AND of its included literals (an empty clause is 1 while training, 0 at test)
  * vote_k = #positive clauses firing - #negative clauses firing, clipped to [-T, T]
  * training sample (x, y): class y clauses get feedback with probability (T - v_y)/2T
    (Type I to positive, Type II to negative); one random other class q gets feedback with
    probability (T + v_q)/2T (Type II to positive, Type I to negative)
  * Type I : clause = 1 -> literal 1: +1 with prob (s-1)/s, literal 0: -1 with prob 1/s
             clause = 0 -> every literal: -1 with prob 1/s
  * Type II: clause = 1 -> every excluded literal that is 0: +1
  * class = argmax of the votes (ties: lowest index)

Split, per held-out subject:
  train        trials 1-4 of the training subjects (TRAIN_SAMPLES windows drawn at random)
  validation   trial 5 of the training subjects: keeps the best epoch
  calibration  trial CALIB_TRIAL of the held-out subject
  test         the held-out subject's other trials (the same for every row)
Calibration = CALIB_EPOCHS more epochs of the same TM rules on the calibration windows only
(counters and a random-number generator: possible on the chip).
The INT8 LDA (without and with the calibration trial, refitted offline) is the reference.
Prints results only; writes no files.

Run:  python K_4_TM_Calib.py
      (K_4_HDC_Way1_vs_Way2.py, K_4_LDA_Analysis.py and lda_golden_model.py in the same folder;
       it reuses the data loading of K_4_HDC_Way1_vs_Way2.py)
"""

import time
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import K_4_HDC_Way1_vs_Way2 as H          # data loading and statistics
import lda_golden_model as G

# ─── Settings ─────────────────────────────────────────────────────────────────
BITS          = 4         # thermometer thresholds per feature
CLAUSES       = 100       # per class (half positive, half negative)
T             = 50        # vote target
S             = 5.0       # specificity
N_STATES      = 100       # automaton has 2 * N_STATES states
EPOCHS        = 8
TRAIN_SAMPLES = 10000     # windows per epoch, drawn at random from the training set
CALIB_TRIAL   = 1
CALIB_EPOCHS  = 10
SEED          = 0

C, D, K = H.C, H.D, H.K
POL = np.where(np.arange(CLAUSES) < CLAUSES // 2, 1, -1)
POS, NEG = POL > 0, POL < 0


# ─── Booleanisation ───────────────────────────────────────────────────────────
def thresholds(xq_train):
    qs = np.linspace(0, 100, BITS + 2)[1:-1]
    return np.percentile(xq_train, qs, axis=0).T                     # (D, BITS)


def literals(xq, th):
    bits = (xq[:, :, None] >= th[None]).reshape(len(xq), -1)        # (n, D*BITS)
    return np.concatenate([bits, ~bits], axis=1)                     # (n, L) bool


# ─── Tsetlin machine ──────────────────────────────────────────────────────────
class TM:
    def __init__(self, n_lit, rng):
        self.rng = rng
        # start every automaton at the boundary, on either side
        self.state = rng.choice([N_STATES, N_STATES + 1], size=(K, CLAUSES, n_lit)).astype(np.int16)

    def copy(self):
        t = TM.__new__(TM)
        t.rng, t.state = self.rng, self.state.copy()
        return t

    def _clauses(self, k, x):
        inc = self.state[k] > N_STATES
        return ~(inc & ~x).any(axis=1)                                # empty clause -> 1 (training)

    def _type1(self, k, rows, out, x):
        if not rows.any():
            return
        st = self.state[k, rows]
        o = out[rows][:, None]
        r = self.rng.random(st.shape)
        up = o & x & (r < (S - 1) / S)
        down = ((o & ~x) | ~o) & (r < 1 / S)
        st += up.astype(np.int16) - down.astype(np.int16)
        np.clip(st, 1, 2 * N_STATES, out=st)
        self.state[k, rows] = st

    def _type2(self, k, rows, out, x):
        rows = rows & out
        if not rows.any():
            return
        st = self.state[k, rows]
        st += (~x & (st <= N_STATES)).astype(np.int16)
        self.state[k, rows] = st

    def update(self, x, y):
        out = self._clauses(y, x)
        v = np.clip((out * POL).sum(), -T, T)
        sel = self.rng.random(CLAUSES) < (T - v) / (2 * T)
        self._type1(y, sel & POS, out, x)
        self._type2(y, sel & NEG, out, x)

        q = self.rng.integers(K - 1)
        q += q >= y
        out = self._clauses(q, x)
        v = np.clip((out * POL).sum(), -T, T)
        sel = self.rng.random(CLAUSES) < (T + v) / (2 * T)
        self._type2(q, sel & POS, out, x)
        self._type1(q, sel & NEG, out, x)

    def fit(self, X, y, epochs, n_per_epoch=None, Xva=None, yva=None):
        best = (-1.0, self.state.copy())
        for _ in range(epochs):
            n = len(X) if n_per_epoch is None else min(n_per_epoch, len(X))
            for i in self.rng.permutation(len(X))[:n]:
                self.update(X[i], y[i])
            if Xva is not None:
                acc = np.mean(self.predict(Xva) == yva)
                if acc > best[0]:
                    best = (acc, self.state.copy())
        if Xva is not None:
            self.state = best[1]
        return self

    def predict(self, X):
        inc = (self.state > N_STATES).reshape(K * CLAUSES, -1).astype(np.float32)
        viol = (~X).astype(np.float32) @ inc.T                        # included literals that are 0
        fire = (viol == 0) & (inc.sum(axis=1) > 0)                    # empty clause -> 0 (test)
        votes = (fire.reshape(len(X), K, CLAUSES) * POL).sum(axis=2)
        return np.argmax(votes, axis=1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"  FORS-EMG — Tsetlin machine + per-user calibration (trial {CALIB_TRIAL}), LOSO")
    print(f"  {BITS} bits/feature, {CLAUSES} clauses/class, T = {T}, s = {S}, {2 * N_STATES} states")
    print("=" * 78)
    recs, ids = H.load()
    raw, y, sid, trial = [], [], [], []
    for q8, lbl, s, t in tqdm(recs, desc="Features"):
        r, full = G.window_features(q8, C)
        r = r[full]
        raw.append(r); n = len(r)
        y.append(np.full(n, lbl)); sid.append(np.full(n, s)); trial.append(np.full(n, t))
    raw, y, sid, trial = np.vstack(raw), np.concatenate(y), np.concatenate(sid), np.concatenate(trial)
    del recs

    keys = ["lda", "lda_c", "tm", "tm_c"]
    res = {k: [] for k in keys}
    rng = np.random.default_rng(SEED)
    for i, s in enumerate(ids):
        tr = (sid != s) & np.isin(trial, H.TRAIN_TRIALS)
        va = (sid != s) & (trial == H.VAL_TRIAL)
        ca = (sid == s) & (trial == CALIB_TRIAL)
        te = (sid == s) & (trial != CALIB_TRIAL)

        mu, k = G.calibrate_norm(raw[tr | va])
        xq = G.normalize(raw, mu, k)

        def lda_acc(mask):
            lda, W, b = G.fit_int8_lda(xq[mask], y[mask])
            return np.mean(lda.classes_[np.argmax(xq[te] @ W.T + b, axis=1)] == y[te])

        res["lda"].append(lda_acc(tr | va))
        res["lda_c"].append(lda_acc(tr | va | ca))

        X = literals(xq, thresholds(xq[tr]))
        tm = TM(X.shape[1], rng).fit(X[tr], y[tr], EPOCHS, TRAIN_SAMPLES, X[va], y[va])
        res["tm"].append(np.mean(tm.predict(X[te]) == y[te]))
        tmc = tm.copy().fit(X[ca], y[ca], CALIB_EPOCHS)
        res["tm_c"].append(np.mean(tmc.predict(X[te]) == y[te]))

        print(f"  [{i + 1:2d}/{len(ids)}] S{s:02d}   LDA {res['lda'][-1]:.1%} -> {res['lda_c'][-1]:.1%}   "
              f"TM {res['tm'][-1]:.1%} -> {res['tm_c'][-1]:.1%}   [{(time.time() - t0) / 60:.1f} min]",
              flush=True)

    print(f"\n{'─' * 78}\n  Test accuracy on trials != {CALIB_TRIAL} (mean over {len(ids)} subjects; ± = 95% CI)\n{'─' * 78}")
    rows = []
    for name, a, b in (("INT8 LDA", "lda", "lda_c"), ("Tsetlin machine", "tm", "tm_c")):
        ma, ha = H.mean_ci(res[a]); mb, hb = H.mean_ci(res[b])
        rows.append([name, f"{ma * 100:.2f} ± {ha * 100:.2f}", f"{mb * 100:.2f} ± {hb * 100:.2f}",
                     f"{(mb - ma) * 100:+.2f}"])
    print(tabulate(rows, headers=["Model", "No calibration %", "With calibration %", "Gain pp"],
                   tablefmt="simple"))
    print()
    print(tabulate([["TM calib vs TM"] + H.paired(res["tm_c"], res["tm"]),
                    ["TM vs LDA"] + H.paired(res["tm"], res["lda"]),
                    ["TM calib vs LDA (no calib)"] + H.paired(res["tm_c"], res["lda"]),
                    ["TM calib vs LDA calib"] + H.paired(res["tm_c"], res["lda_c"])],
                   headers=["Comparison", "Δ pp", "95% CI", "Wilcoxon p", "Up/Down", "Verdict"],
                   tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time() - t0) / 60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
