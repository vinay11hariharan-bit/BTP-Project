"""
Bit-exact golden model of the EMG LDA core (Stages 1-3).

Two implementations of the same integer arithmetic:
  LdaCoreRef      - sample-by-sample, mirrors the RTL counters/registers (use for Verilog test vectors)
  window_features - vectorized over a whole recording (use for training on the full dataset)
main() cross-checks them, so training data and hardware are guaranteed to see identical features.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

N_CLASSES = 6                       # classes evaluated by the hardware
N_TYPES = 5                         # per channel, in this order: SABS, WL, ZC, SSC, POW
SUB_WIN = 98                        # samples per bank = stride; window = 2 * SUB_WIN = 196
BANK_BITS = (14, 15, 7, 7, 21)      # per-bank field widths (64 bits per channel per bank)
INT32_MIN, INT32_MAX = -2**31, 2**31 - 1


# ---------------------------------------------------------------- arithmetic shared by both models

def to_adc_int8(emg, full_scale):
    """Model the 8-bit ADC: electrode voltage -> signed INT8. full_scale = input range referred to electrode."""
    return np.clip(np.round(emg / full_scale * 128), -128, 127).astype(np.int8)


def mitchell_sq(a):
    """Mitchell-1 square of a in [0, 128]: (a << (p+1)) - (1 << 2p), p = leading-one position; 0 -> 0."""
    if a == 0:
        return 0
    p = a.bit_length() - 1
    return (a << (p + 1)) - (1 << (2 * p))


POW_LUT = np.array([mitchell_sq(a) for a in range(129)], dtype=np.int64)


def normalize(raw, mu, k):
    """Shift-normalizer: clip(((raw - mu) + 2^(k-1)) >> k, -128, 127), arithmetic shift, k >= 1."""
    return np.clip(((raw - mu) + (np.int64(1) << (k - 1))) >> k, -128, 127)


# ---------------------------------------------------------------- cycle-faithful reference (RTL mirror)

class LdaCoreRef:
    def __init__(self, C, mu, k, W, b):
        self.C, self.D = C, N_TYPES * C
        self.mu, self.k, self.W, self.b = mu, k, W, b
        self.x_prev = [0] * C                                   # hist_ram: x[n-1]
        self.d0 = [0] * C                                       # hist_ram: previous difference
        self.bank = [[[0] * N_TYPES for _ in range(C)] for _ in range(2)]   # feat_acc_ram
        self.bank_sel = 0
        self.cnt_win = 0
        self.strides = 0
        self.cycles = 0

    def step(self, sample):
        """One ADC frame (C signed samples). Returns None, or a decision dict after every 98th sample."""
        C, D = self.C, self.D
        start = self.cycles

        # Stage 1: feature ALU, one channel per cycle
        for ch in range(C):
            x, xp, d0 = int(sample[ch]), self.x_prev[ch], self.d0[ch]
            a, d1 = abs(x), x - xp
            feats = (a,                                         # SABS
                     abs(d1),                                   # WL
                     int((x < 0) != (xp < 0)),                  # ZC  (MSB xor)
                     int((d1 > 0 and d0 < 0) or (d1 < 0 and d0 > 0)),  # SSC
                     mitchell_sq(a))                            # POW
            row = self.bank[self.bank_sel][ch]
            for t in range(N_TYPES):
                row[t] = feats[t] if self.cnt_win == 0 else row[t] + feats[t]
                assert row[t] < 2 ** BANK_BITS[t], "bank field overflow"
            self.x_prev[ch], self.d0[ch] = x, d1
        self.cycles += C

        if self.cnt_win < SUB_WIN - 1:
            self.cnt_win += 1
            return None
        self.cnt_win = 0
        self.strides += 1

        # Stage 2: bank combine + shift-normalize; nested counters ch (outer) x type (inner)
        q = np.zeros(D, dtype=np.int64)
        for ch in range(C):
            for t in range(N_TYPES):
                f = N_TYPES * ch + t
                raw = self.bank[0][ch][t] + self.bank[1][ch][t]
                q[f] = normalize(np.int64(raw), self.mu[f], self.k[f])
        self.cycles += D
        self.bank_sel ^= 1

        # Stage 3: bias preload + D MACs per class, streaming argmax, output strobe
        scores, best, max_score = [], 0, 0
        for cls in range(N_CLASSES):
            acc = int(self.b[cls])
            for f in range(D):
                acc += int(q[f]) * int(self.W[cls, f])
                assert INT32_MIN <= acc <= INT32_MAX, "accumulator overflow"
            if cls == 0 or acc > max_score:
                max_score, best = acc, cls
            scores.append(acc)
        self.cycles += N_CLASSES * (D + 1) + 1

        return dict(cls=best, scores=scores, q=q, full=self.strides >= 2,
                    peak_cycles=self.cycles - start)


# ---------------------------------------------------------------- vectorized equivalent (for training)

def window_features(rec, C):
    """Stages 1-2 for a whole recording (N x C int8). Returns raw window sums (n_strides x D) and full-window flags."""
    x = rec.astype(np.int64)
    zero = np.zeros((1, C), np.int64)
    x_prev = np.vstack([zero, x[:-1]])
    d1 = x - x_prev
    d0 = np.vstack([zero, d1[:-1]])
    a = np.abs(x)
    feats = np.stack([a,
                      np.abs(d1),
                      ((x < 0) != (x_prev < 0)).astype(np.int64),
                      (((d1 > 0) & (d0 < 0)) | ((d1 < 0) & (d0 > 0))).astype(np.int64),
                      POW_LUT[a]], axis=-1)                                # (N, C, 5)
    S = len(x) // SUB_WIN
    sub = feats[:S * SUB_WIN].reshape(S, SUB_WIN, C, N_TYPES).sum(axis=1)  # one bank per stride
    assert S == 0 or np.all(sub.max(axis=(0, 1)) < 2 ** np.array(BANK_BITS)), "bank field overflow"
    prev = np.concatenate([np.zeros((1, C, N_TYPES), np.int64), sub[:-1]])
    raw = (sub + prev).reshape(S, C * N_TYPES)                             # channel-major order
    return raw, np.arange(S) >= 1


# ---------------------------------------------------------------- calibration and INT8 LDA

def calibrate_norm(raw):
    """mu_f = rounded mean; k_f = smallest shift >= 1 that never clips on the calibration set."""
    mu = np.round(raw.mean(axis=0)).astype(np.int64)
    dev = np.abs(raw - mu).max(axis=0)
    k = np.maximum(1, np.ceil(np.log2(np.maximum(dev, 1) / 127))).astype(np.int64)
    assert mu.max() < 2**22 and k.max() < 2**5
    return mu, k


def quantize_lda(coef, intercept):
    """Float linear form (n_cls x D, n_cls) -> INT8 weights with one scale s_w, INT32 bias on the same scale."""
    n = len(intercept)
    assert coef.shape[0] == n and 3 <= n <= N_CLASSES
    s_w = np.abs(coef).max() / 127
    W = np.round(coef / s_w).astype(np.int64)
    b = np.round(intercept / s_w).astype(np.int64)
    assert np.abs(b).max() <= INT32_MAX
    # unused hardware classes: zero weights, bias INT32_MIN -> can never win the strict '>' argmax
    W = np.vstack([W, np.zeros((N_CLASSES - n, W.shape[1]), np.int64)])
    b = np.concatenate([b, np.full(N_CLASSES - n, INT32_MIN, np.int64)])
    return W, b


def fit_int8_lda(q, y):
    """Fit LDA on hardware features q and quantize it."""
    lda = LinearDiscriminantAnalysis().fit(q, y)
    return (lda, *quantize_lda(lda.coef_, lda.intercept_))


def dataset_features(recordings, C):
    """Full windows only, from a list of (int8 recording N x C, label)."""
    R, Y = [], []
    for rec, label in recordings:
        raw, full = window_features(rec, C)
        R.append(raw[full])
        Y += [label] * int(full.sum())
    return np.vstack(R), np.array(Y)


# ---------------------------------------------------------------- demo / self-check

def main(C=8, seed=0):
    # Replace these synthetic recordings with real ones: a list of (int8 array N x C, label) per trial,
    # e.g. to_adc_int8(trial_volts, full_scale). For LOSO, calibrate and fit on training subjects only.
    rng = np.random.default_rng(seed)
    gains = rng.uniform(4, 40, size=(N_CLASSES, C))

    def recording(cls):
        return to_adc_int8(gains[cls] * rng.standard_normal((10 * SUB_WIN, C)), 128.0)

    train = [(recording(c), c) for c in range(N_CLASSES) for _ in range(20)]
    test = [(recording(c), c) for c in range(N_CLASSES) for _ in range(5)]

    raw_tr, y_tr = dataset_features(train, C)
    mu, k = calibrate_norm(raw_tr)
    lda, W, b = fit_int8_lda(normalize(raw_tr, mu, k), y_tr)

    raw_te, y_te = dataset_features(test, C)
    q_te = normalize(raw_te, mu, k)
    scores = q_te @ W.T + b
    assert scores.min() >= INT32_MIN and scores.max() <= INT32_MAX
    pred_int = lda.classes_[np.argmax(scores, axis=1)]      # np.argmax keeps the first max = strict '>'
    pred_float = lda.predict(q_te)

    # cross-check: cycle-faithful reference vs vectorized model on one recording
    rec, _ = test[0]
    core = LdaCoreRef(C, mu, k, W, b)
    decisions = [d for s in rec if (d := core.step(s)) is not None]
    raw, _ = window_features(rec, C)
    q_fast = normalize(raw, mu, k)
    for i, d in enumerate(decisions):
        assert np.array_equal(d["q"], q_fast[i])
        assert d["scores"] == [int(v) for v in q_fast[i] @ W.T + b]

    D = N_TYPES * C
    print(f"C = {C}, D = {D}")
    print(f"  accuracy float LDA {np.mean(pred_float == y_te):.3f} | INT8 core {np.mean(pred_int == y_te):.3f}"
          f" | agreement {np.mean(pred_int == pred_float):.3f}")
    print(f"  reference == vectorized on {len(decisions)} decisions: OK")
    print(f"  peak cycles {decisions[-1]['peak_cycles']} (36C+7 = {36 * C + 7})"
          f" | cycles/decision {core.cycles // len(decisions)} (133C+7 = {133 * C + 7})")
    print(f"  k_f range {k.min()}..{k.max()} | W range {W.min()}..{W.max()} | max |score| {np.abs(scores).max()}")
    d = decisions[1]                                          # first full window: a ready-made test vector
    print(f"  test vector (window 2): class {d['cls']}, scores {d['scores']}")
    print(f"  x_q = {d['q'].tolist()}")


if __name__ == "__main__":
    for C in (8, 24):
        main(C)
