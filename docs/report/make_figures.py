"""Figures for the report (PDF for LaTeX, PNG at 300 dpi). Run from the repository root:
    python docs/report/make_figures.py
The data figures re-run the chip-model LOSO on archive/ (a few minutes); the other figures use the
numbers printed by the scripts listed in the report's appendix."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "golden_model"), str(ROOT / "analysis")]
import numpy as np
import K_4_LDA_Golden_LOSO as GL
from K_4_LDA_Golden_LOSO import B, G
from scipy.signal import welch, sosfreqz
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "legend.fontsize": 7, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
INK, ACC, MUTE, TEAL, GOLD = "#1f3b4d", "#c0392b", "#9aa8b1", "#1a8a7a", "#d4a017"
GEST = ["Hand Close", "Hand Open", "Wrist Flexion", "Wrist Extension"]
GCOL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
FS = 985.0
OUT = None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=INK, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=6, color=INK)


# --------------------------------------------------------------------------- diagrams
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 1.55)); ax.axis("off"); ax.set_xlim(0, 14.4); ax.set_ylim(0, 2.2)
    steps = [("8-24 forearm\nelectrodes", "#eceff1"), ("20-450 Hz BP\n+ 50 Hz notch\n(causal)", "#fdebd0"),
             ("8-bit ADC\n985 Hz", "#fdebd0"), ("Stage 1\n5 integer features\nper channel", "#d1f2eb"),
             ("Stage 2\n$(x-\\mu)\\gg k$\nclip to INT8", "#d1f2eb"), ("Stage 3\nINT8 LDA\n1 MAC, argmax", "#d1f2eb"),
             ("gesture\nevery 99.5 ms", "#eceff1")]
    w, g = 1.68, 0.36
    for i, (t, c) in enumerate(steps):
        x = 0.1 + i * (w + g)
        box(ax, x, 0.55, w, 1.1, t, c)
        if i:
            arrow(ax, x - g, 1.1, x, 1.1)
    ax.text(0.1 + 3 * (w + g), 0.2, "196-sample window, 98-sample stride", fontsize=6.5, color=TEAL)
    ax.text(0.1 + 1 * (w + g), 0.2, "analog / ADC pod", fontsize=6.5, color="#b9770e")
    ax.text(0.1 + 3 * (w + g) + 2 * (w + g) + w, 1.9, "chip, clk_sys = 1 MHz", fontsize=6.5, color=TEAL, ha="right")
    save(fig, "pipeline")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.axis("off"); ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.2)
    ax.add_patch(Rectangle((2.55, 0.25), 2.5, 5.6, fc="#fef5e7", ec="#b9770e", lw=0.7, ls="--"))
    ax.text(3.8, 5.6, "clk_spi  8 MHz", ha="center", fontsize=7, color="#b9770e")
    ax.add_patch(Rectangle((5.45, 0.25), 8.8, 5.6, fc="#e8f8f5", ec=TEAL, lw=0.7, ls="--"))
    ax.text(9.85, 5.6, "clk_sys  1 MHz", ha="center", fontsize=7, color=TEAL)
    for i in range(3):
        box(ax, 0.1, 3.9 - i * 1.3, 2.0, 0.95, f"ADC pod {i}\n8 channels", "#eceff1")
    arrow(ax, 1.1, 2.6, 1.1, 2.55); arrow(ax, 1.1, 3.9, 1.1, 3.85)
    arrow(ax, 2.1, 4.35, 2.8, 4.35, "MISO")
    box(ax, 2.8, 3.8, 2.0, 1.1, "emg_spi_adc\nCONVST, DRDY,\nburst of 8C bits", "#fdebd0")
    box(ax, 2.8, 1.9, 2.0, 1.0, "cdc_toggle_sync\n2-FF + edge", "#fdebd0")
    arrow(ax, 3.8, 3.8, 3.8, 2.9, "frame_tog")
    box(ax, 5.8, 3.8, 2.4, 1.1, "Stage 1\nemg_feat_accum\nhistory + 2 banks", "#d1f2eb")
    arrow(ax, 4.8, 4.35, 5.8, 4.35, "rx_flat")
    arrow(ax, 4.8, 2.4, 6.4, 3.8)
    box(ax, 8.8, 3.8, 2.3, 1.1, "Stage 2\nemg_shift_norm\n24-bit datapath", "#d1f2eb")
    arrow(ax, 8.2, 4.35, 8.8, 4.35, "bank words")
    box(ax, 11.7, 3.8, 2.3, 1.1, "x_q RAM\n120 x 8 bit", "#eceff1")
    arrow(ax, 11.1, 4.35, 11.7, 4.35, "$x_q$")
    box(ax, 11.7, 1.9, 2.3, 1.2, "Stage 3\nemg_lda_mac\n8x8 MAC, INT32,\nstreaming argmax", "#d1f2eb")
    arrow(ax, 12.85, 3.8, 12.85, 3.1)
    box(ax, 8.8, 1.9, 2.3, 1.2, "emg_cfg_regs\n$\\mu, k, W, b$,\ncfg_n_ch, core_en", "#f2f3f4")
    arrow(ax, 11.1, 2.5, 11.7, 2.5, "W, b")
    arrow(ax, 9.95, 3.1, 9.95, 3.8, "$\\mu, k$")
    box(ax, 5.8, 1.9, 2.4, 1.2, "emg_core_ctrl\nstride_done,\nbank swap, primed", "#f2f3f4")
    arrow(ax, 7.0, 3.8, 7.0, 3.1)
    box(ax, 11.7, 0.45, 2.3, 0.9, "class_valid\nclass_out[2:0]", "#fadbd8")
    arrow(ax, 12.85, 1.9, 12.85, 1.35)
    box(ax, 8.8, 0.45, 2.3, 0.9, "host cfg port\nwe, addr, wdata", "#eceff1")
    arrow(ax, 9.95, 1.35, 9.95, 1.9)
    save(fig, "architecture")


def fig_banks():
    fig, ax = plt.subplots(figsize=(7.2, 1.9)); ax.set_yticks([]); ax.grid(False)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    for i in range(6):
        ax.add_patch(Rectangle((i * 98, 1.6), 98, 0.5, fc=["#aed6f1", "#f5cba7"][i % 2], ec=INK, lw=0.6))
        ax.text(i * 98 + 49, 1.85, f"bank {i % 2}", ha="center", va="center", fontsize=6.5)
    for j in range(4):
        y = 1.1 - j * 0.32
        ax.add_patch(Rectangle(((j + 1) * 98 - 98, y), 196, 0.22, fc=GCOL[j % 4], alpha=0.35, ec=GCOL[j % 4]))
        ax.text((j + 1) * 98 + 98 + 6, y + 0.11, f"window {j + 1} -> decision", va="center", fontsize=6.5)
        ax.plot([(j + 2) * 98] * 2, [y, 1.6], color=ACC, lw=0.7, ls=":")
    ax.set_xlim(0, 640); ax.set_ylim(-0.1, 2.2); ax.set_xlabel("sample index (985 Hz)")
    ax.set_title("Ping-pong banks: a window is bank 0 + bank 1; Stages 2-3 run after every 98th sample")
    save(fig, "banks")


# --------------------------------------------------------------------------- data figures
def fig_signals(d):
    lbl = list(d["ex_lbl"]); raw, flt = d["ex_raw"], d["ex_f"]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 2.9), sharex=True)
    t = np.arange(raw.shape[1]) / FS
    for c, g in enumerate(range(4)):
        i = lbl.index(g)
        axs[0, c].plot(t, raw[i][:, 0], lw=0.3, color=MUTE); axs[0, c].set_title(GEST[g], color=GCOL[g])
        for ch in range(8):
            axs[1, c].plot(t, flt[i][:, ch] + ch * 0.6, lw=0.25, color=GCOL[g])
        axs[1, c].set_xlabel("time (s)"); axs[1, c].set_yticks([])
    axs[0, 0].set_ylabel("raw, ch 1"); axs[1, 0].set_ylabel("filtered, ch 1-8")
    save(fig, "signals")


def fig_spectrum(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    axs[0].semilogy(d["f"], d["P0"], color=MUTE, lw=0.8, label="raw")
    axs[0].semilogy(d["f"], d["P1"], color=ACC, lw=0.8, label="filtered (causal)")
    axs[0].axvline(50, color="k", lw=0.5, ls=":"); axs[0].text(53, d["P0"].max() / 3, "50 Hz", fontsize=6.5)
    axs[0].set_xlabel("frequency (Hz)"); axs[0].set_ylabel("PSD (Welch, mean over ch.)"); axs[0].legend(frameon=False)
    axs[0].set_title("Spectrum, subject 2, all recordings")
    axs[1].plot(d["w"], 20 * np.log10(np.maximum(d["h"], 1e-6)), color=INK, lw=0.9)
    axs[1].axvspan(20, 450, color=TEAL, alpha=0.08); axs[1].set_ylim(-80, 5)
    axs[1].set_xlabel("frequency (Hz)"); axs[1].set_ylabel("|H| (dB)"); axs[1].set_title("4th-order BP 20-450 Hz + notch Q=30")
    save(fig, "spectrum")


def fig_gesture_sets():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    labels = ["A: 4 incl.\nIndex", "B: 4\nselected", "D: best\n6 of 12", "C: all\n12"]
    acc, chance = [65.28, 79.38, 69.0, 48.8], [25, 25, 16.7, 8.3]
    x = np.arange(4)
    ax.bar(x, acc, color=[MUTE, ACC, INK, INK], width=0.6, zorder=2)
    ax.scatter(x, chance, marker="_", s=300, color="k", zorder=3, label="chance")
    for i, a in enumerate(acc):
        ax.text(i, a + 1.5, f"{a:.1f}", ha="center", fontsize=7)
    ax.annotate("+14.1 pp, p = 3.8e-9", (0.75, 82), (-0.35, 88), fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("INT8 LOSO accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right"); ax.set_title("Gesture vocabulary (19 subjects)")
    save(fig, "gesture_sets")


def fig_mlp_widths():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    names = ["64-4", "64-16-4", "64-32-4", "64-64-4", "64-128-4"]
    b = [272, 1168, 2320, 4624, 9232]; f = [65.17, 63.71, 63.89, 64.80, 64.22]; q = [65.28, 63.63, 63.93, 64.87, 64.02]
    ax.errorbar(b, q, yerr=5.9 / 2, fmt="o", color=INK, ms=3, capsize=2, lw=0.6, label="INT8 (bar: half 95% CI)")
    ax.scatter(b, f, marker="x", color=ACC, s=14, label="FP32", zorder=3)
    for n, x, y in zip(names, b, q):
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.set_xscale("log"); ax.set_ylim(56, 70); ax.set_xlabel("model memory (bytes)"); ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=6); ax.set_title("MLP width sweep, Set A: all tied")
    save(fig, "mlp_widths")


CLF = [("MLP 64-16-4", 79.65, 79.57, 1088, 1168), ("Hist. grad. boost", 79.63, 79.26, 0, 19200),
       ("Logistic reg.", 79.33, 79.19, 256, 272), ("Linear SVM", 79.31, 78.30, 256, 272),
       ("LDA", 77.91, 77.87, 256, 272), ("Random forest", 77.84, 77.43, 0, 53145),
       ("Extra trees", 76.29, 76.04, 0, 38015), ("Decision tree", 73.69, 73.17, 0, 5055)]


def fig_classifiers():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]; y = np.arange(len(CLF))[::-1]
    for yy, (n, f, q, m, b) in zip(y, CLF):
        c = ACC if n == "LDA" else (MUTE if b > 10240 else INK)
        ax.barh(yy, q, color=c, height=0.6, zorder=2); ax.plot(f, yy, "k|", ms=8, zorder=3)
        ax.text(q - 0.3, yy, f"{q:.2f}", va="center", ha="right", color="w", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in CLF]); ax.set_xlim(70, 81)
    ax.set_xlabel("accuracy (%)  (bar: INT8, tick: FP32)"); ax.set_title("Eight classifiers, Set B (19 subjects)")
    ax = axs[1]
    for n, f, q, m, b in CLF:
        c = ACC if n == "LDA" else (MUTE if b > 10240 else INK)
        ax.scatter(b, q, color=c, s=20, zorder=3)
        ax.annotate(n, (b, q), xytext=(4, -3), textcoords="offset points", fontsize=6)
    ax.axvline(10240, color="k", lw=0.6, ls="--"); ax.text(11500, 73.4, "10 KB budget", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("model memory (bytes)"); ax.set_ylabel("INT8 accuracy (%)"); ax.set_ylim(72.5, 80.5)
    ax.set_title("Accuracy vs memory")
    save(fig, "classifiers")


def fig_qdrop():
    pts = [(c[1], c[2]) for c in CLF] + list(zip([65.17, 63.71, 63.89, 64.80, 64.22], [65.28, 63.63, 63.93, 64.87, 64.02])) \
        + [(79.69, 79.38), (48.99, 48.37), (49.71, 49.28), (69.23, 69.21), (68.95, 68.74)]
    f, q = np.array(pts).T
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    axs[0].plot([45, 82], [45, 82], color=MUTE, lw=0.6); axs[0].scatter(f, q, s=12, color=INK, zorder=3)
    axs[0].set_xlabel("FP32 accuracy (%)"); axs[0].set_ylabel("INT8 accuracy (%)"); axs[0].set_title("INT8 vs FP32 (line: equal)")
    axs[1].hist(f - q, bins=np.arange(-0.3, 1.11, 0.1), color=INK, rwidth=0.85, zorder=2)
    axs[1].axvline(0.18, color=ACC, lw=0.8); axs[1].text(0.21, axs[1].get_ylim()[1] * 0.85, "mean of all 28: 0.18 pp", color=ACC, fontsize=6.5)
    axs[1].set_xlabel("INT8 loss, FP32 - INT8 (pp)"); axs[1].set_ylabel("comparisons"); axs[1].set_title(f"Quantisation loss ({len(f)} listed in Tab. 4)")
    save(fig, "qdrop")


SP = np.array([[1, .998, .978, .192, .331, .998, .077], [.998, 1, .977, .188, .325, 1, .088],
               [.978, .977, 1, .095, .198, .977, .046], [.192, .188, .095, 1, .713, .188, .3],
               [.331, .325, .198, .713, 1, .325, .215], [.998, 1, .977, .188, .325, 1, .088],
               [.077, .088, .046, .3, .215, .088, 1]])
FT = ["MAV", "RMS", "WL", "ZC", "SSC", "VAR", "WAMP"]


def fig_featsel():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    im = axs[0].imshow(SP, cmap="Blues", vmin=0, vmax=1); axs[0].grid(False)
    axs[0].set_xticks(range(7)); axs[0].set_xticklabels(FT, rotation=90, fontsize=6); axs[0].set_yticks(range(7)); axs[0].set_yticklabels(FT, fontsize=6)
    for i in range(7):
        for j in range(7):
            axs[0].text(j, i, f"{SP[i, j]:.2f}".lstrip("0") if SP[i, j] < 1 else "1", ha="center", va="center", fontsize=5,
                        color="w" if SP[i, j] > .6 else "k")
    axs[0].set_title("Spearman |rho| between types")
    k = np.arange(1, 8); acc = [72.10, 74.51, 76.08, 76.70, 76.99, 76.99, 76.92]
    axs[1].plot(k, acc, "o-", color=INK, ms=3); axs[1].plot(5, 76.99, "o", color=ACC, ms=6)
    axs[1].set_xlabel("feature types k"); axs[1].set_ylabel("best inner accuracy (%)"); axs[1].set_title("Accuracy vs k (nested LOSO)")
    fr = [18, 18, 18, 17, 14, 1, 0]; nm = ["MAV", "ZC", "VAR", "WL", "SSC", "RMS", "WAMP"]
    axs[2].barh(range(7)[::-1], np.array(fr) / 18 * 100, color=[ACC] * 5 + [MUTE] * 2, zorder=2)
    axs[2].set_yticks(range(7)[::-1]); axs[2].set_yticklabels(nm); axs[2].set_xlabel("selected in % of 18 folds")
    axs[2].set_title("Selection frequency")
    save(fig, "feature_selection")


def fig_mitchell():
    a = np.arange(0, 128)
    p = np.floor(np.log2(np.maximum(a, 1))).astype(int)
    m1 = np.where(a > 0, a * 2 ** (p + 1) - 2 ** (2 * p), 0)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1))
    axs[0].plot(a, a ** 2, color=MUTE, lw=1.2, label="$a^2$"); axs[0].plot(a, m1, color=ACC, lw=0.8, label="Mitchell-1")
    axs[0].set_xlabel("|x| (ADC code)"); axs[0].set_ylabel("square"); axs[0].legend(frameon=False); axs[0].set_title("Exact and approximate square")
    r = np.where(a > 0, (a ** 2 - m1) / np.maximum(a ** 2, 1) * 100, 0)
    axs[1].plot(a[1:], r[1:], color=INK, lw=0.8); axs[1].set_xlabel("|x| (ADC code)"); axs[1].set_ylabel("underestimate (%)")
    axs[1].set_title("Never over; 0 at powers of 2, at most 25%")
    save(fig, "mitchell")


H = {"R": [68.9, 65, 76.77, 68.69, 87.33, 71.12, 81.12, 80.27, 81.71, 85.69, 82.54, 63.08, 84.1, 59.08, 71.56, 91.79, 83.88, 91.4],
     "H6": [68.79, 64.08, 76.75, 68.52, 87.08, 71.4, 80.75, 80.38, 81.25, 85.81, 82.67, 62.62, 83.62, 59.52, 71.35, 91.9, 83.46, 91.23],
     "H5": [67.02, 56.79, 75.65, 65.56, 85.23, 70.56, 81.69, 78.54, 79.17, 82.52, 80.12, 62.73, 79.46, 57.4, 69.9, 90.73, 83.08, 90.71]}


def fig_hw_variants():
    names = ["H1 exact", "H2 Mitchell-2", "H3 Mitchell-1", "H4 (S|x|)^2", "H5 no power", "H6 chosen"]
    d = [0.19, 0.11, -0.05, -0.31, -1.91, -0.05]; lo = [-0.36, -0.41, -0.57, -0.88, -2.94, -0.59]; hi = [0.73, 0.63, 0.47, 0.27, -0.87, 0.48]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
    y = np.arange(6)[::-1]
    for yy, n, m, l, h in zip(y, names, d, lo, hi):
        c = ACC if "H6" in n else (MUTE if "H5" in n else INK)
        axs[0].plot([l, h], [yy, yy], color=c, lw=1.4); axs[0].plot(m, yy, "o", color=c, ms=4)
    axs[0].axvline(0, color="k", lw=0.5); axs[0].axvline(-1, color=ACC, lw=0.6, ls="--"); axs[0].text(-0.95, 0.3, "NI margin", fontsize=6, color=ACC)
    axs[0].set_yticks(y); axs[0].set_yticklabels(names); axs[0].set_xlabel("paired diff. vs float reference (pp, 95% CI)")
    axs[0].set_title("Integer feature sets, shift normaliser")
    r = np.array(H["R"])
    for k, c in [("H6", ACC), ("H5", MUTE)]:
        axs[1].scatter(r, np.array(H[k]) - r, s=12, color=c, label=k, zorder=3)
    axs[1].axhline(0, color="k", lw=0.5); axs[1].set_xlabel("float reference accuracy per subject (%)")
    axs[1].set_ylabel("variant - reference (pp)"); axs[1].legend(frameon=False); axs[1].set_title("Per subject (std. normaliser)")
    save(fig, "hw_variants")


def fig_norm(d):
    q = d["q"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1))
    axs[0].hist(q.ravel(), bins=np.arange(-128.5, 128.5, 4), color=INK, zorder=2)
    axs[0].set_yscale("log"); axs[0].set_xlabel("$x_q$ (INT8)"); axs[0].set_ylabel("count")
    clip = np.mean((q == 127) | (q == -128)) * 100
    axs[0].set_title(f"Normalised features, held-out subjects (clipped {clip:.2f}%)")
    k = d["k"].reshape(-1, 5)
    im = axs[1].imshow(k.T, aspect="auto", cmap="YlGnBu"); axs[1].grid(False)
    axs[1].set_yticks(range(5)); axs[1].set_yticklabels(["SABS", "WL", "ZC", "SSC", "POW"]); axs[1].set_xlabel("channel")
    axs[1].set_xticks(range(k.shape[0])); axs[1].set_xticklabels(range(1, k.shape[0] + 1))
    for i in range(k.shape[0]):
        for j in range(5):
            axs[1].text(i, j, int(k[i, j]), ha="center", va="center", fontsize=6)
    axs[1].set_title("Shift $k_f$ per feature (one fold)")
    save(fig, "normaliser")


def fig_lda(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for g in range(4):
        m = d["pytr"] == g
        axs[0].scatter(*d["ptr"][m].T, s=1, alpha=0.15, color=GCOL[g])
        m = d["pyte"] == g
        axs[0].scatter(*d["pte"][m][::3].T, s=2, marker="x", alpha=0.6, color=GCOL[g], label=GEST[g])
    axs[0].set_xlabel("LD 1"); axs[0].set_ylabel("LD 2"); axs[0].legend(frameon=False, markerscale=4, fontsize=6)
    axs[0].set_title(f"LDA space: training (dots), held-out S{int(d['psid']):02d} (x)")
    W = d["W"]; v = np.abs(W).max()
    axs[1].imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); axs[1].grid(False)
    axs[1].set_yticks(range(4)); axs[1].set_yticklabels(GEST, fontsize=6.5)
    axs[1].set_xticks(np.arange(2, 40, 5)); axs[1].set_xticklabels([f"ch{c}" for c in range(1, 9)], fontsize=6)
    for c in range(1, 8):
        axs[1].axvline(c * 5 - 0.5, color="k", lw=0.3)
    axs[1].set_title("INT8 weights $W$ (5 features per channel)")
    save(fig, "lda")


def fig_confusion(d):
    cm = d["cm"]; n = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.9)); ax.grid(False)
    ax.imshow(n, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{n[i, j]:.1f}", ha="center", va="center", color="w" if n[i, j] > 50 else "k", fontsize=7)
    ab = ["HC", "HO", "WF", "WE"]
    ax.set_xticks(range(4)); ax.set_xticklabels(ab); ax.set_yticks(range(4)); ax.set_yticklabels(ab)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Chip model, all held-out windows\n(row %, accuracy {np.trace(cm) / cm.sum() * 100:.1f}%)")
    save(fig, "confusion")


G_CHIP = [66.52, 67.81, 74.5, 71.81, 84.5, 72.12, 82.33, 80.17, 81.21, 84.1, 82.6, 70.92, 81.04, 59.42, 71.46, 91.44, 83.62, 91.48]
G_REF = [68.1, 68.02, 74.6, 73.71, 82.6, 71.71, 82.08, 80.6, 81.02, 85.56, 83.21, 70.52, 81.6, 61.33, 71.4, 90.75, 82.56, 91.46]


def fig_per_subject(d):
    sid = [f"S{i:02d}" for i in range(2, 20)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[2, 1]))
    x = np.arange(18)
    axs[0].bar(x - 0.2, G_REF, 0.4, color=MUTE, label="software INT8 LDA", zorder=2)
    axs[0].bar(x + 0.2, G_CHIP, 0.4, color=ACC, label="chip (golden model)", zorder=2)
    axs[0].plot(x, d["tracc"] * 100, "_", color="k", ms=7, label="chip, training subjects")
    axs[0].set_xticks(x); axs[0].set_xticklabels(sid, rotation=90, fontsize=6); axs[0].set_ylim(50, 100)
    axs[0].axhline(25, color="k", lw=0.4); axs[0].set_ylabel("accuracy (%)"); axs[0].legend(frameon=False, ncol=3, fontsize=6, loc="upper left")
    axs[0].set_title("Per held-out subject (LOSO, 18 subjects)")
    axs[1].plot([55, 95], [55, 95], color=MUTE, lw=0.6); axs[1].scatter(G_REF, G_CHIP, s=12, color=ACC, zorder=3)
    axs[1].set_xlabel("software INT8 LDA (%)"); axs[1].set_ylabel("chip (%)")
    axs[1].set_title("diff -0.21 pp [-0.72, +0.30]")
    save(fig, "per_subject")


def fig_cycles():
    C = np.array([8, 12, 16, 20, 24]); D = 5 * C
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    s2 = D + 2; s3 = 6 * (D + 1) + 2
    axs[0].bar(C, s2, width=2.6, color=TEAL, label="Stage 2: D+2", zorder=2)
    axs[0].bar(C, s3, width=2.6, bottom=s2, color=INK, label="Stage 3: 6(D+1)+2", zorder=2)
    axs[0].plot(C, [290, 430, 570, 710, 850], "o", color=ACC, ms=4, label="measured (RTL)", zorder=3)
    axs[0].axhline(1015, color="k", ls="--", lw=0.8); axs[0].text(26, 1045, "1015 cycles per sample", ha="right", fontsize=6.5)
    axs[0].set_xticks(C); axs[0].set_xlabel("channels C"); axs[0].set_ylabel("clk_sys cycles"); axs[0].set_ylim(0, 1250)
    axs[0].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0, 0.8)); axs[0].set_title("Stages 2+3 after each stride")
    mem = [("feature banks", 2 * 24 * 64), ("weights", 720 * 8), ("$\\mu, k$", 120 * 27), ("$x_q$", 120 * 8),
           ("history", 24 * 10), ("biases", 6 * 32)]
    n, v = zip(*mem)
    axs[1].barh(range(6)[::-1], np.array(v) / 1000, color=INK, zorder=2)
    for i, val in enumerate(v):
        axs[1].text(val / 1000 + 0.05, 5 - i, f"{val} bit", va="center", fontsize=6.5)
    axs[1].set_yticks(range(6)[::-1]); axs[1].set_yticklabels(n); axs[1].set_xlabel("kbit"); axs[1].set_xlim(0, 7.5)
    axs[1].set_title(f"Memory, sized for C=24 (total {sum(v) / 1000:.1f} kbit)")
    save(fig, "cycles")


def fig_verification():
    fig, ax = plt.subplots(figsize=(7.2, 1.9)); ax.axis("off"); ax.set_xlim(0, 14.4); ax.set_ylim(0, 2.6)
    box(ax, 0.1, 1.3, 2.4, 1.0, "FORS-EMG\ntraining subjects", "#eceff1")
    box(ax, 3.0, 1.3, 2.6, 1.0, "golden model\nbit-exact features,\nfit INT8 LDA", "#fdebd0")
    box(ax, 6.1, 1.3, 2.4, 1.0, "cfg port writes\n$\\mu, k, W, b$", "#f2f3f4")
    box(ax, 9.0, 1.3, 2.4, 1.0, "RTL emg_top\n(Icarus + cocotb)", "#d1f2eb")
    box(ax, 6.1, 0.05, 2.4, 0.9, "ADC daisy-chain\nmodel, gesture stream", "#eceff1")
    box(ax, 11.9, 1.3, 2.4, 1.0, "compare per stride:\n$x_q$ (D), 6 scores,\nclass", "#fadbd8")
    arrow(ax, 2.5, 1.8, 3.0, 1.8); arrow(ax, 5.6, 1.8, 6.1, 1.8); arrow(ax, 8.5, 1.8, 9.0, 1.8); arrow(ax, 11.4, 1.8, 11.9, 1.8)
    arrow(ax, 8.5, 0.5, 10.2, 1.3)
    ax.annotate("", (13.1, 2.3), (4.3, 2.3), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, connectionstyle="arc3,rad=-0.12"))
    ax.text(8.7, 2.48, "expected values from the same golden model", ha="center", fontsize=6.5)
    ax.text(13.1, 0.6, "pass: C = 8, 12, 16,\n20, 24 bit-exact", ha="center", fontsize=6.5, color=TEAL)
    save(fig, "verification")


def plot_all(d, out):
    global OUT
    OUT = out
    fig_pipeline(); fig_architecture(); fig_banks(); fig_gesture_sets(); fig_mlp_widths(); fig_classifiers()
    fig_qdrop(); fig_featsel(); fig_mitchell(); fig_hw_variants(); fig_cycles(); fig_verification()
    if d is not None:
        fig_signals(d); fig_spectrum(d); fig_norm(d); fig_lda(d); fig_confusion(d); fig_per_subject(d)


def compute():
    """Filter the dataset, run the chip-model LOSO and collect what the data figures need."""
    recs, ids, fs_ = GL.load_recordings()
    subjects = B.load_all_subjects(B.BASE_DIR)
    # example raw vs filtered (first subject, one trial each gesture, Rest)
    s0=subjects[0]; ex={}
    filt=B._design_filters()
    for rec,lbl,m in zip(s0["recordings"],s0["labels"],s0["meta"]):
        if m["trial"]==1 and m["orientation"]=="Rest" and lbl not in ex:
            ex[lbl]=(rec, B.apply_preprocessing_filter(rec.T,filt,"causal").T)
    # PSD before/after over subject 0
    P0=[];P1=[]
    for rec in s0["recordings"]:
        f,p=welch(rec.T,fs=B.FS,nperseg=512); P0.append(p.mean(0))
        _,p=welch(B.apply_preprocessing_filter(rec.T,filt,"causal"),fs=B.FS,nperseg=512); P1.append(p.mean(0))
    w,h=sosfreqz(filt["sos"],worN=4096,fs=B.FS)
    chip,ref,probe,_=GL.build_features(recs,fs_)
    C=GL.C; cm=np.zeros((4,4),int); per={}; tracc={}; proj=None; qall=[]
    for t in sorted(set(chip["sid"])):
        tr=(chip["sid"]!=t)&np.isin(chip["trial"],GL.TRAIN_TRIALS); te=chip["sid"]==t
        mu,k=G.calibrate_norm(chip["raw"][tr]); qtr=G.normalize(chip["raw"][tr],mu,k); qte=G.normalize(chip["raw"][te],mu,k)
        lda=B.build_lda().fit(qtr,chip["y"][tr]); Wf,bf=GL.linear_form(lda); W,b=G.quantize_lda(Wf,bf)
        pr=lda.classes_[np.argmax(qte@W.T+b,1)]; y=chip["y"][te]
        for a,c in zip(y,pr): cm[a,c]+=1
        per[t]=np.mean(pr==y); tracc[t]=np.mean(lda.classes_[np.argmax(qtr@W.T+b,1)]==chip["y"][tr])
        qall.append(qte)
        if proj is None:
            v=L(solver="eigen",shrinkage="auto").fit(qtr,chip["y"][tr])
            proj=dict(tr=v.transform(qtr)[::7,:2],ytr=chip["y"][tr][::7],te=v.transform(qte)[:,:2],yte=y,W=W[:4],mu=mu,k=k,sid=t)
    return dict(ex_lbl=np.array(list(ex)), ex_raw=np.array([ex[k][0] for k in ex]), ex_f=np.array([ex[k][1] for k in ex]),
                f=f, P0=np.mean(P0, 0), P1=np.mean(P1, 0), w=w, h=np.abs(h), cm=cm, tracc=np.array(list(tracc.values())),
                ptr=proj["tr"], pytr=proj["ytr"], pte=proj["te"], pyte=proj["yte"], W=proj["W"], k=proj["k"], psid=proj["sid"],
                q=np.vstack(qall)[::5])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    plot_all(compute(), out)
    print("figures in", out)
