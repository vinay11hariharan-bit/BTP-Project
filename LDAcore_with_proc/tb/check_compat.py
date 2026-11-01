"""check_compat.py - proves the new lda_golden_model.py (layer over golden_model.py, legacy preset)
gives exactly the same numbers as the old lda_golden_model.py (lda_golden_model_original.py)."""
import os

import numpy as np

os.environ.update(EMG_BANK_LEN="98", EMG_N_BANKS="2", EMG_WARMUP="1", EMG_REQUANT="old",
                  EMG_CHIP_LDA="1", EMG_DEAD_RULE="wl_int8")               # the old chip
import lda_golden_model as NEW
import lda_golden_model_original as OLD


def run(C, seed):
    rng = np.random.default_rng(seed)
    gains = rng.uniform(4, 40, size=(6, C))
    rec = lambda c, n: OLD.to_adc_int8(gains[c] * rng.standard_normal((n * 98, C)), 128.0)
    train = [(rec(c, 10), c) for c in range(6) for _ in range(10)]
    test = [(rec(c, 8), c) for c in range(6) for _ in range(2)]
    for name in ("SUB_WIN", "N_CLASSES", "N_TYPES", "BANK_BITS", "INT32_MIN", "INT32_MAX"):
        assert getattr(NEW, name) == getattr(OLD, name), name
    assert np.array_equal(NEW.POW_LUT, OLD.POW_LUT)
    codes = rng.integers(-(1 << 23), 1 << 23, size=5000)
    for sh in range(0, 17):
        a, b = NEW.ads_code_to_int8(codes, sh), OLD.ads_code_to_int8(codes, sh)
        assert a.dtype == b.dtype and np.array_equal(a, b)
    for x in range(-128, 128, 7):
        for xp in range(-128, 128, 11):
            for d0 in (-3, 0, 5):
                assert NEW.sample_features(x, xp, d0) == OLD.sample_features(x, xp, d0)
    rn, yn = NEW.dataset_features(train, C)
    ro, yo = OLD.dataset_features(train, C)
    assert np.array_equal(rn, ro) and np.array_equal(yn, yo)
    for r, _ in test:
        a, b = NEW.window_features(r, C), OLD.window_features(r, C)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    mn, kn = NEW.calibrate_norm(rn)
    mo, ko = OLD.calibrate_norm(ro)
    assert np.array_equal(mn, mo) and np.array_equal(kn, ko)
    qn, qo = NEW.normalize(rn, mn, kn), OLD.normalize(ro, mo, ko)
    assert np.array_equal(qn, qo)
    _, Wn, bn = NEW.fit_int8_lda(qn, yn)
    _, Wo, bo = OLD.fit_int8_lda(qo, yo)
    assert np.array_equal(Wn, Wo) and np.array_equal(bn, bo)
    n = 0
    for r, _ in test:
        cn, co = NEW.LdaCoreRef(C, mn, kn, Wn, bn), OLD.LdaCoreRef(C, mo, ko, Wo, bo)
        for s in r:
            dn, do = cn.step(s), co.step(s)
            assert (dn is None) == (do is None)
            if dn is not None:
                assert set(do) <= set(dn)                  # new fields: valid, dead
                for key in do:
                    assert np.array_equal(dn[key], do[key]), key
                n += 1
    return n


if __name__ == "__main__":
    tot = sum(run(C, s) for C in (8, 16, 24) for s in range(3))
    print(f"compat: new lda_golden_model == old on every function, C = 8/16/24, 3 seeds, {tot} decisions")
