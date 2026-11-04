"""
FORS-EMG — cost-weighted correlated best-arm identification (CT-BAI) as a gesture decision rule
==============================================================================================
Binary gesture decision = two-armed Gaussian best-arm identification:
  arm k        gesture k of the pair (k = 1, 2)
  arm mean     E[S_k], the LDA discriminant of gesture k under the gesture being held;
               for Gaussian class models with shared covariance, argmax_k E[S_k | y] = y
  joint pull   one new non-overlapping window (196 samples) -> (U, V) = (S_1, S_2) on it
  correlation  rho = corr(S_1, S_2) on the same window; unknown for a new user (Case II)
  cost         one window of latency; c = c_12 (both scores come from the same window), so the
               cost-optimal allocation is pure joint sampling and every pull is a joint pull
Stopping rule (Compilation Document): Z_t = profiled GLR of Proposition 4.1 with N1 = N2 = 0,
threshold beta(t, delta) of Eq. (32) (Cauchy-Uniform prior), stop at the first t with
Z_t > beta(t, delta), recommend the arm with the larger mean.

Gap with the paper's model: the paper assumes unit variances. Here each score is divided by
one common scale (pooled within-class std from the training subjects), which keeps the order
of the means but leaves the two variances only approximately 1. Whether delta still holds on
real users is what the "error" column measures.

Second rule from the paper (Theorem 5.2): the mixture statistic itself,
  W_t = log( integral of L_t dP ) - sup over Alt of L_t,   P = Cauchy x Cauchy x Uniform prior (30),
is delta-PAC with the constant threshold log(1/delta) (Ville's inequality), no curvature term.
The integral is computed numerically: for each rho, the mean integral is a Gaussian in
(mu1, mu2) times the Cauchy prior, done by Gauss-Hermite quadrature; rho is integrated on a
fine grid in z = atanh(rho) (uniform prior on rho, Jacobian 1 - rho^2).

Compared rules (same windows; the three GLR rules share the threshold beta(t, delta)):
  CT-BAI mixture  W_t > log(1/delta)               (Theorem 5.2)
  CT-BAI GLR      Z_t > beta(t, delta), rho unknown, profiled (Proposition 4.1 + Eq. 32)
  GLR rho = 0     correlation ignored:            Z_t = t (U_bar - V_bar)^2 / 4
  GLR rho_train   rho fixed from training users:  Z_t = t (U_bar - V_bar)^2 / (4 (1 - rho_train))
  1 window        today's chip: sign of S_1 - S_2 on the first window
  vote of n       majority of the first n windows

Each FORS-EMG recording starts at rest: windows 0-15 (about 1.5 s) are at chance level, so
every rule starts at window START_WINDOW, as the chip would after its rest flag clears.
Data: x_q from the bit-exact Stage 1 + 2 model, LOSO as in the other scripts (normaliser and
LDA fitted on the training subjects only), every recording of the held-out subject is one
decision, for all 6 gesture pairs. Prints results only; writes no files.

Run:  python K_2_CTBAI_Stopping.py
      (K_4_HDC_Way1_vs_Way2.py, K_4_LDA_Analysis.py and lda_golden_model.py in the same folder)
"""

import itertools
import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabulate import tabulate
from tqdm import tqdm

import K_4_HDC_Way1_vs_Way2 as H          # data loading
import K_4_LDA_Analysis as B
import lda_golden_model as G

DELTAS = (0.1, 0.05, 0.01)
START_WINDOW = 20         # skip rest / gesture onset (~2 s): the arm means must be stationary
VOTES = (3, 5)
WINDOW_MS = 2 * G.SUB_WIN / B.FS * 1000
RHO_CLIP = 0.999
C = H.C


# ─── The paper's statistic, joint pulls only (N1 = N2 = 0) ────────────────────
def loglik(r, n, suu, svv, suv):
    """l_t at correlation r, with the means already profiled out (residual sums given)."""
    return -0.5 * n * np.log(1 - r * r) - (suu - 2 * r * suv + svv) / (2 * (1 - r * r))


def rho_mle(n, suu, svv, suv):
    """Root of score equation (III) with the means profiled: the cubic
    n r (1 - r^2) - r (suu + svv) + (1 + r^2) suv = 0, i.e.
    -n r^3 + suv r^2 + (n - suu - svv) r + suv = 0; best root in (-1, 1)."""
    roots = np.roots([-n, suv, n - suu - svv, suv])
    cand = [float(np.clip(x.real, -RHO_CLIP, RHO_CLIP)) for x in roots if abs(x.imag) < 1e-9]
    cand += [-RHO_CLIP, RHO_CLIP]
    return max(cand, key=lambda r: loglik(r, n, suu, svv, suv))


def ctbai_stat(U, V):
    """(Z_t, mu1_hat, mu2_hat, rho_hat) after t = len(U) joint pulls."""
    n = len(U)
    ub, vb = U.mean(), V.mean()
    du, dv = U - ub, V - vb
    suu, svv, suv = du @ du, dv @ dv, du @ dv
    r = rho_mle(n, suu, svv, suv)                                  # unconstrained: means = U_bar, V_bar
    m = (ub + vb) / 2                                             # Eq. (21) with N1 = N2 = 0
    cu, cv = U - m, V - m
    r_alt = rho_mle(n, cu @ cu, cv @ cv, cu @ cv)                 # constrained, mu1' = mu2' = m
    z = loglik(r, n, suu, svv, suv) - loglik(r_alt, n, cu @ cu, cv @ cv, cu @ cv)
    return z, ub, vb, r, suu / n, svv / n


# Mixture over the prior (Theorem 5.2), joint pulls only
_Z = np.linspace(-6, 6, 601)                               # z = atanh(rho)
_R = np.tanh(_Z)
_LOGW_R = np.log(0.5 * (1 - _R ** 2) * (_Z[1] - _Z[0]))     # uniform prior on rho, Jacobian, step
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(16)      # nodes for E over N(0, 1)
_GH_W = _GH_W / _GH_W.sum()


def log_cauchy(x):
    return -np.log(np.pi) - np.log1p(x * x)


def mixture_logint(U, V):
    """log of the integral of L_t over the Cauchy-Cauchy-Uniform prior (same l_t as Eq. (18),
    without the 2 pi constant, which cancels against the sup)."""
    n = len(U)
    ub, vb = U.mean(), V.mean()
    du, dv = U - ub, V - vb
    suu, svv, suv = du @ du, dv @ dv, du @ dv
    r = _R
    # l_t(mu, rho) = -n/2 log(1-r^2) - [suu - 2 r suv + svv]/(2(1-r^2)) - n/2 d' S^-1 d,  d = (ub-mu1, vb-mu2)
    base = -0.5 * n * np.log(1 - r * r) - (suu - 2 * r * suv + svv) / (2 * (1 - r * r))
    # integral over mu of exp(-n/2 d' S^-1 d) p(mu1) p(mu2) = (2 pi sqrt(1-r^2) / n) E_{mu ~ N(mean, S/n)}[p p]
    a = np.sqrt(1.0 / n)
    x1 = ub + a * _GH_X                                        # mu1 = ub + a z1
    # mu2 = vb + a (r z1 + sqrt(1-r^2) z2)   (Cholesky of S = [[1, r], [r, 1]])
    z1, z2 = np.meshgrid(_GH_X, _GH_X, indexing="ij")
    w2 = np.outer(_GH_W, _GH_W)
    lp1 = log_cauchy(ub + a * z1)                               # (q, q)
    mu2 = vb + a * (r[:, None, None] * z1 + np.sqrt(1 - r * r)[:, None, None] * z2)
    lp = lp1[None] + log_cauchy(mu2)                           # (nr, q, q)
    m = lp.max(axis=(1, 2), keepdims=True)
    log_e = np.log(np.sum(w2[None] * np.exp(lp - m), axis=(1, 2))) + m[:, 0, 0]
    log_mu = np.log(2 * np.pi) + 0.5 * np.log(1 - r * r) - np.log(n) + log_e
    tot = base + log_mu + _LOGW_R
    mx = tot.max()
    return mx + np.log(np.exp(tot - mx).sum())


def mixture_stat(U, V):
    """W_t = log int L_t dP - sup_{Alt} l_t (constrained MLE as in Section 4.2)."""
    n = len(U)
    m = (U.mean() + V.mean()) / 2
    cu, cv = U - m, V - m
    r_alt = rho_mle(n, cu @ cu, cv @ cv, cu @ cv)
    return mixture_logint(U, V) - loglik(r_alt, n, cu @ cu, cv @ cv, cu @ cv)


def beta(t, delta, mu1, mu2, rho, s2u, s2v):
    """Eq. (32) with M_t of Definition 5.3 (joint pulls only: U_bar - mu1_hat = 0)."""
    eps0 = (1 - rho * rho) / 2
    d_mean = 1.0
    k_local = max(s2u, s2v) + 1.0
    M = 16 * k_local / eps0 ** 3 + (8 * d_mean + 2) / eps0 ** 2 + 1
    return (np.log(1 / delta) + 1.5 * np.log(t) + 1.5 * np.log(M)
            + np.log(1 + mu1 * mu1) + np.log(1 + mu2 * mu2) + 3)


def run_rules(U, V, rho_train, delta):
    """Returns {rule: (decision in {0, 1} or None, windows used)}; decision 0 = arm 1."""
    out = {k: (None, len(U)) for k in ("CT-BAI mixture", "CT-BAI GLR", "GLR rho = 0", "GLR rho_train")}
    for t in range(2, len(U) + 1):
        z, mu1, mu2, r, s2u, s2v = ctbai_stat(U[:t], V[:t])
        b = beta(t, delta, mu1, mu2, r, s2u, s2v)
        dec = int(mu2 > mu1)
        gap2 = (mu1 - mu2) ** 2
        if out["CT-BAI mixture"][0] is None and mixture_stat(U[:t], V[:t]) > np.log(1 / delta):
            out["CT-BAI mixture"] = (dec, t)
        for name, stat in (("CT-BAI GLR", z),
                           ("GLR rho = 0", t * gap2 / 4),
                           ("GLR rho_train", t * gap2 / (4 * (1 - rho_train)))):
            if out[name][0] is None and stat > b:
                out[name] = (dec, t)
        if all(v[0] is not None for v in out.values()):
            break
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 84)
    print("  FORS-EMG — CT-BAI stopping rules (Thm 5.2 mixture; Prop. 4.1 + Eq. 32) for gesture decisions, LOSO")
    print("=" * 84)
    recs, ids = H.load()
    raw, y, sid, trial, rid = [], [], [], [], []
    for j, (q8, lbl, s, t) in enumerate(tqdm(recs, desc="Features")):
        r, full = G.window_features(q8, C)
        r = r[full]
        n = len(r)
        raw.append(r); y.append(np.full(n, lbl)); sid.append(np.full(n, s))
        trial.append(np.full(n, t)); rid.append(np.full(n, j))
    raw, y, sid, trial, rid = (np.vstack(raw), np.concatenate(y), np.concatenate(sid),
                               np.concatenate(trial), np.concatenate(rid))
    del recs
    names = B.GESTURE_NAMES

    rules = ["CT-BAI mixture", "CT-BAI GLR", "GLR rho = 0", "GLR rho_train"]
    acc = {(p, d, k): [] for p in itertools.combinations(range(H.K), 2) for d in DELTAS for k in rules}
    fixed = {(p, k): [] for p in itertools.combinations(range(H.K), 2)
             for k in ["1 window"] + [f"vote of {n}" for n in VOTES]}
    rho_seen = {p: ([], []) for p in itertools.combinations(range(H.K), 2)}

    for s in tqdm(ids, desc="Held-out subjects"):
        tr = sid != s
        mu, k = G.calibrate_norm(raw[tr])
        xq = G.normalize(raw, mu, k).astype(float)
        for pair in itertools.combinations(range(H.K), 2):
            m = tr & np.isin(y, pair)
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", store_covariance=True)
            lda.fit(xq[m], y[m])
            Sig, means = lda.covariance_, lda.means_                       # rows in order of pair
            W = np.linalg.solve(Sig, means.T).T                             # w_k = Sigma^-1 mu_k
            b = -0.5 * np.sum(W * means, axis=1)                            # equal priors
            cov = W @ Sig @ W.T
            scale = np.sqrt(np.trace(cov) / 2)                              # common scale: keeps the order
            rho_train = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            S_tr = (xq[m] @ W.T + b) / scale
            centre = S_tr.mean()

            for r_id in np.unique(rid[(sid == s) & np.isin(y, pair)]):
                w = np.where(rid == r_id)[0][START_WINDOW::2]              # held gesture, non-overlapping windows
                S = (xq[w] @ W.T + b) / scale - centre
                U, V = S[:, 0], S[:, 1]
                truth = pair.index(y[w[0]])
                rho_seen[pair][0].append(np.corrcoef(U, V)[0, 1]); rho_seen[pair][1].append(rho_train)

                dec1 = (V > U).astype(int)
                fixed[(pair, "1 window")].append((dec1[0] == truth, 1))
                for n in VOTES:
                    fixed[(pair, f"vote of {n}")].append((np.bincount(dec1[:n], minlength=2).argmax() == truth, n))
                for d in DELTAS:
                    res = run_rules(U, V, rho_train, d)
                    for name in rules:
                        dec, used = res[name]
                        acc[(pair, d, name)].append((None if dec is None else dec == truth, used))

    def summarise(items):
        decided = [c for c, _ in items if c is not None]
        err = 100 * (1 - np.mean(decided)) if decided else float("nan")
        undecided = 100 * (1 - len(decided) / len(items))
        n_used = np.mean([u for c, u in items if c is not None]) if decided else float("nan")
        return err, undecided, n_used

    pairs = list(itertools.combinations(range(H.K), 2))
    print(f"\n  {len(ids)} held-out subjects, all 6 gesture pairs; one decision per recording.")
    print(f"  A pull = one non-overlapping window = {WINDOW_MS:.0f} ms; rules start at window {START_WINDOW} (after onset).\n")

    print("─" * 84 + "\n  Correlation between the two scores on the same window (the joint pull's rho)\n" + "─" * 84)
    print(tabulate([[f"{names[a]} vs {names[b_]}", f"{np.mean(rho_seen[(a, b_)][1]):+.2f}",
                     f"{np.mean(rho_seen[(a, b_)][0]):+.2f}",
                     f"{np.percentile(rho_seen[(a, b_)][0], 10):+.2f} .. {np.percentile(rho_seen[(a, b_)][0], 90):+.2f}"]
                    for a, b_ in pairs],
                   headers=["Pair", "rho from training", "rho per recording (mean)", "10-90% range"],
                   tablefmt="simple"))

    print("\n" + "─" * 84 + "\n  All pairs pooled\n" + "─" * 84)
    rows = []
    for k in ["1 window"] + [f"vote of {n}" for n in VOTES]:
        items = sum((fixed[(p, k)] for p in pairs), [])
        e, u, n = summarise(items)
        rows.append([k, "", f"{e:.2f}", f"{u:.1f}", f"{n:.2f}", f"{n * WINDOW_MS:.0f}"])
    for d in DELTAS:
        for name in rules:
            items = sum((acc[(p, d, name)] for p in pairs), [])
            e, u, n = summarise(items)
            flag = "  <= delta" if e <= 100 * d else "  > delta"
            rows.append([name, f"{d}", f"{e:.2f}{flag}", f"{u:.1f}", f"{n:.2f}", f"{n * WINDOW_MS:.0f}"])
    print(tabulate(rows, headers=["Rule", "delta", "Error % (decided)", "Undecided %", "Windows", "Latency ms"],
                   tablefmt="simple"))

    print("\n" + "─" * 84 + f"\n  Per pair, delta = {DELTAS[1]}\n" + "─" * 84)
    rows = []
    for p in pairs:
        row = [f"{names[p[0]]} vs {names[p[1]]}"]
        e, _, _ = summarise(fixed[(p, "1 window")]); row.append(f"{e:.1f}")
        for name in rules:
            e, u, n = summarise(acc[(p, DELTAS[1], name)])
            row.append(f"{e:.1f} / {n:.1f} / {u:.0f}")
        rows.append(row)
    print(tabulate(rows, headers=["Pair", "1 window err %"] + [f"{r}: err % / windows / undec %" for r in rules],
                   tablefmt="simple"))
    print(f"\n  Total runtime : {(time.time() - t0) / 60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
