"""Waveform pictures of the chip-level simulation (GTKWave style, PNG).

Run the chip test first (python tb/test_emg_top.py with CHANNELS = 8), then, from RTL/:
    python tb/plot_waveforms.py
Reads waveform/emg_top_all.vcd and writes ../docs/waveforms/*.png. The times of the views
are those of the C = 8 run (seed 8).
"""
import bisect, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VCD = HERE.parent / "waveform" / "emg_top_all.vcd"
OUTDIR = HERE.parents[1] / "docs" / "waveforms"

def parse(vcd):
    want={"emg_top":["adc_drdy_n","adc_miso","arst_n","cfg_addr","cfg_wdata","cfg_we","clk_spi","clk_sys","xq_we","xq_raddr","xq_q","xq_data","xq_addr","w_re","w_data","w_addr","swap","stride_done","score_valid","score_cls","score","sample_done","primed","overrun","norm_start","norm_mu","norm_k","norm_done","norm_busy","norm_addr","mac_start","mac_busy","frame_tog","frame_start","core_en","class_valid","class_out","cfg_n_ch","busy1","bank_sel","b_data","adc_sclk","adc_cs_n","adc_convst","rst_sys_n","rst_spi_n","s2_addr","s2_re"],
     "u_accum":["rd_ch","wr_ch","wr_x","wr_valid","alu_sabs","alu_wl","alu_zc","alu_ssc","alu_pow","cnt_win","bank0_we","bank1_we","acc_sabs","acc_pow","first_sample"],
     "u_adc":["state","bit_idx","byte_idx","timer"],"u_cdc":["s1","s2","s3"],"u_mitchell_sq":["a","p","sq"]}
    ids={}; scope=[]
    f=open(vcd)
    for line in f:
        t=line.split()
        if not t: continue
        if t[0]=="$scope": scope.append(t[2])
        elif t[0]=="$upscope": scope.pop()
        elif t[0]=="$var":
            sc=scope[-1]; name=t[4]
            if sc in want and name in want[sc]:
                ids.setdefault(t[3],[]).append((f"{sc}.{name}",int(t[2])))
        elif t[0]=="$enddefinitions": break
    sig=collections.defaultdict(list); width={}
    for k,v in ids.items():
        for n,w in v: width[n]=w
    T=0
    for line in f:
        c=line[0]
        if c=="#": T=int(line[1:]); continue
        if c in "01xzXZ":
            k=line[1:].strip()
            if k in ids:
                for n,_ in ids[k]: sig[n].append((T,line[0]))
        elif c=="b":
            v,k=line[1:].split()
            if k in ids:
                for n,_ in ids[k]: sig[n].append((T,v))
    return dict(sig), width, T



S = WID = None
BG, GRID, TRACE, BUS, TXT, NAME, MARK = "#000000", "#2a2a2a", "#00ff00", "#00e0ff", "#ffff60", "#ffffff", "#ff4040"
SIGNED = {"score", "xq_data", "xq_q", "w_data", "wr_x", "b_data"}


def fmt(name, v, w):
    if any(c in v for c in "xzXZ"):
        return "x"
    n = int(v, 2)
    base = name.split(".")[-1]
    if base in SIGNED and w > 1 and n >= 1 << (w - 1) and len(v) >= w:
        n -= 1 << w
    if base in ("cfg_addr",):
        return f"{n:03X}"
    if base in ("cfg_wdata",):
        return f"{n:X}"
    return str(n)


def changes(name, t0, t1):
    ch = S[name]
    times = [c[0] for c in ch]
    i = max(bisect.bisect_right(times, t0) - 1, 0)
    out = [(t0, ch[i][1])]
    for t, v in ch[i + 1:]:
        if t > t1:
            break
        out.append((t, v))
    return out


def draw(fname, title, rows, t0_us, t1_us, markers=(), unit="us", h_row=0.42, width=13.0, label_w=2.3):
    t0, t1 = int(t0_us * 1e6), int(t1_us * 1e6)
    n = len(rows)
    H = 1.0 + h_row * n
    fig = plt.figure(figsize=(width, H), facecolor=BG)
    ax = fig.add_axes([label_w / width, 0.25 / H, 1 - label_w / width - 0.01, (h_row * n) / H])
    ax.set_facecolor(BG)
    scale = 1e6 if unit == "us" else 1e9
    x0, x1 = t0 / scale, t1 / scale
    ax.set_xlim(x0, x1); ax.set_ylim(-n, 0.2)
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    ax.tick_params(colors=NAME, labelsize=8, top=True, labeltop=True, bottom=False, labelbottom=False, left=False, labelleft=False)
    for sp in ax.spines.values():
        sp.set_color("#555555")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.text(0.005, 1 - 0.22 / H, f"{title}    (time in {'µs' if unit == 'us' else 'ms'})", color=NAME, fontsize=10, family="monospace", va="center")
    for r, (name, label) in enumerate(rows):
        y = -r - 0.85
        fig.text(0.005, ax.get_position().y0 + (y + n) / (n + 0.2) * ax.get_position().height + 0.35 / (n + 0.2) * ax.get_position().height,
                 label, color=NAME, fontsize=8.5, family="monospace", va="center")
        ax.axhline(y - 0.12, color="#1a1a1a", lw=0.5)
        w = WID[name]
        ch = changes(name, t0, t1)
        pts = [(t / scale, v) for t, v in ch] + [(x1, None)]
        if w == 1:
            dense = len(ch) > 600
            if dense:
                ax.fill_between([x0, x1], y, y + 0.6, color=TRACE, alpha=0.35, lw=0)
                continue
            xs, ys = [], []
            for (ta, va), (tb, _) in zip(pts[:-1], pts[1:]):
                lv = y + (0.6 if va == "1" else 0.0)
                if va in "xXzZ":
                    ax.fill_between([ta, tb], y, y + 0.6, color=MARK, alpha=0.5, lw=0)
                xs += [ta, tb]; ys += [lv, lv]
            ax.plot(xs, ys, color=TRACE, lw=1.0, drawstyle="default")
        else:
            span = x1 - x0
            for (ta, va), (tb, _) in zip(pts[:-1], pts[1:]):
                d = min((tb - ta) * 0.25, span * 0.003)
                xs = [ta, ta + d, tb - d, tb, tb - d, ta + d, ta]
                ys = [y + 0.3, y + 0.6, y + 0.6, y + 0.3, y, y, y + 0.3]
                col = MARK if any(c in va for c in "xzXZ") else BUS
                ax.plot(xs, ys, color=col, lw=0.9)
                txt = fmt(name, va, w)
                if (tb - ta) / span * (width - label_w) > 0.075 * max(len(txt), 1) + 0.02:
                    ax.text((ta + tb) / 2, y + 0.3, txt, color=TXT, fontsize=7, ha="center", va="center", family="monospace", clip_on=True)
    for (tm, lab) in markers:
        ax.axvline(tm, color=MARK, lw=0.8, ls="--")
        ax.text(tm, -n - 0.05, lab, color=MARK, fontsize=7.5, ha="center", va="top", family="monospace")
    fig.savefig(OUTDIR / fname, dpi=200, facecolor=BG)
    plt.close(fig)


T = "emg_top."
A = "u_accum."
if __name__ == "__main__":
    S, WID, _ = parse(str(VCD))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    draw("w1_overview.png", "Whole run, C = 8: configuration, 393 ADC frames, 4 strides, 3 decisions",
         [(T + "arst_n", "arst_n"), (T + "core_en", "core_en"), (T + "cfg_we", "cfg_we"), (T + "adc_convst", "adc_convst"), (T + "frame_start", "frame_start"),
          (T + "bank_sel", "bank_sel"), (T + "stride_done", "stride_done"), (T + "primed", "primed"), (T + "class_valid", "class_valid"), (T + "class_out", "class_out[2:0]")],
         0, 400480, markers=[(100.085, "stride 1 (masked)"), (199.846, "decision 1"), (299.316, "decision 2"), (398.786, "decision 3")], unit="ms")
    draw("w2_config.png", "Configuration through the cfg port, then core_en (288 writes: mu/k, W, b, cfg_n_ch, core_en)",
         [(T + "arst_n", "arst_n"), (T + "rst_sys_n", "rst_sys_n (sync)"), (T + "clk_sys", "clk_sys"), (T + "cfg_we", "cfg_we"), (T + "cfg_addr", "cfg_addr[11:0]"),
          (T + "cfg_wdata", "cfg_wdata[31:0]"), (T + "cfg_n_ch", "cfg_n_ch"), (T + "core_en", "core_en")], 0, 60)
    draw("w2b_config_end.png", "End of configuration: biases, cfg_n_ch = 8, core_en = 1",
         [(T + "clk_sys", "clk_sys"), (T + "cfg_we", "cfg_we"), (T + "cfg_addr", "cfg_addr[11:0]"), (T + "cfg_wdata", "cfg_wdata[31:0]"), (T + "cfg_n_ch", "cfg_n_ch"), (T + "core_en", "core_en")],
         570, 588)
    draw("w3_spi_frame.png", "One ADC frame: CONVST -> DRDY_N -> SPI burst of 64 bits -> frame_tog -> 2-FF sync -> frame_start (clk_sys)",
         [(T + "clk_spi", "clk_spi 8 MHz"), (T + "adc_convst", "adc_convst"), (T + "adc_drdy_n", "adc_drdy_n"), ("u_adc.state", "adc.state"), (T + "adc_cs_n", "adc_cs_n"),
          (T + "adc_sclk", "adc_sclk"), (T + "adc_miso", "adc_miso"), ("u_adc.byte_idx", "adc.byte_idx"), (T + "frame_tog", "frame_tog"), (T + "clk_sys", "clk_sys 1 MHz"),
          ("u_cdc.s1", "cdc.s1"), ("u_cdc.s2", "cdc.s2"), ("u_cdc.s3", "cdc.s3"), (T + "frame_start", "frame_start")], 1600.2, 1623.5)
    draw("w4_stage1.png", "Stage 1 on one sample (C = 8): read history + bank word, feature ALU, write back, one channel per clk_sys",
         [(T + "clk_sys", "clk_sys"), (T + "frame_start", "frame_start"), (T + "busy1", "busy1"), (A + "rd_ch", "rd_ch"), (A + "wr_valid", "wr_valid"), (A + "wr_ch", "wr_ch"),
          (A + "wr_x", "x[n] (INT8)"), (A + "alu_sabs", "|x|"), (A + "alu_wl", "|dx|"), (A + "alu_zc", "zc"), (A + "alu_ssc", "ssc"), ("u_mitchell_sq.p", "mitchell p"),
          (A + "alu_pow", "mitchell |x|^2"), (A + "bank0_we", "bank0_we"), (A + "bank1_we", "bank1_we"), (A + "cnt_win", "cnt_win"), (T + "sample_done", "sample_done")], 2635.0, 2650.0)
    draw("w5_stride.png", "Stride (sample 196): bank swap, Stage 2 normaliser (42 cycles), Stage 3 LDA (6 x 41 cycles), decision",
         [(T + "clk_sys", "clk_sys"), (T + "stride_done", "stride_done"), (T + "swap", "swap"), (T + "bank_sel", "bank_sel"), (T + "norm_start", "norm_start"), (T + "norm_busy", "norm_busy"),
          (T + "xq_we", "xq_we"), (T + "norm_done", "norm_done"), (T + "mac_start", "mac_start"), (T + "mac_busy", "mac_busy"), (T + "score_valid", "score_valid"),
          (T + "score_cls", "score_cls"), (T + "score", "score (INT32)"), (T + "class_valid", "class_valid"), (T + "class_out", "class_out")], 199550, 199850,
         markers=[(199555.0, "stride_done"), (199597.0, "mac_start"), (199846.0, "class_valid")])
    draw("w6_norm.png", "Stage 2 detail: per feature, bank0 + bank1 - mu, round, >>> k, clip -> x_q RAM",
         [(T + "clk_sys", "clk_sys"), (T + "norm_start", "norm_start"), (T + "s2_addr", "bank addr (ch)"), (T + "norm_addr", "feature f"), (T + "norm_mu", "mu_f"), (T + "norm_k", "k_f"),
          (T + "xq_we", "xq_we"), (T + "xq_addr", "xq_addr"), (T + "xq_data", "x_q (INT8)"), (T + "norm_done", "norm_done")], 199554, 199580)
    draw("w7_mac.png", "Stage 3 detail: bias load, then one W x x_q MAC per clk_sys (class 0, first 18 of 40 features)",
         [(T + "clk_sys", "clk_sys"), (T + "mac_start", "mac_start"), (T + "b_data", "bias b (INT32)"), (T + "w_addr", "w_addr"), (T + "w_data", "W (INT8)"), (T + "xq_raddr", "xq_raddr"),
          (T + "xq_q", "x_q (INT8)"), (T + "score_valid", "score_valid"), (T + "score_cls", "score_cls"), (T + "score", "score (INT32)")], 199596.5, 199616)
    print("ok")
