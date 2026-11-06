# Branch `multi_board`: up to 8 channels per board, boards in a chain

Each board is one `emg_top` with one 8-channel pod. Boards are chained; together they make one
LDA classifier for 8, 12, 16 or 24 electrodes. Nothing but the K−1 partial sums crosses between
boards.

**The maths.** Stage 3 of every board holds the K−1 difference rows `W'_c = W_c − W_0`
(c = 1..3 for 4 gestures), restricted to its own features, so it computes partial sums
`p_c = Σ W'_cf q_f`. The head also adds `b'_c = b_c − b_0`. The decision only needs the totals
`D_c = Σ_boards p_c`: class = argmax{0, D_1, D_2, D_3} (strict >), which equals argmax_c s_c of the
single-chip LDA. Stages 1–2 are per channel and need no change.

**What changed from `main`.**

| File | Change |
|---|---|
| `emg_chain_link.v` | **new**: head/tail detection, board id, common enable, barrier, bit-serial adder, head argmax |
| `emg_top.v` | one clock (`clk_spi` removed), `CMAX = 8`, `SAMPLE_DIV = 260`, Stage 3 with `NCLS = 3`, `u_link` added, `cfg_board` + chain ports |
| `emg_cfg_regs.v` | memory depths are parameters (`NF`, `NW`); top passes 40 and 120 |
| `cdc_toggle_sync.v` | **removed**: one clock, so `frame_start` is a toggle edge detect in the top |
| other modules | unchanged (`emg_spi_adc` simply runs on `clk_sys`) |

**Clock.** One 256 kHz clock for everything (SCLK 128 kHz, 984.6 Hz sampling, 260 cycles per sample).
With at most 8 channels the longest per-sample work is Stage 1 + Stage 2 = 8 + 42 cycles; Stage 3
(3 × 41 + 2) and the chain transfer (96 bits + 1 cycle per board) overlap the next samples.

**Chain ports of `emg_top`.**

| Port | Dir | Meaning |
|---|---|---|
| `prs_up_i`, `prs_dn_i` | in | a neighbour is present upstream / downstream (head: nothing upstream, tail: nothing downstream) |
| `id_i` → `id_o` | in/out | board id: head 0, +1 per board; `cfg_board` selects which board a cfg write is for |
| `run_i` → `run_o` | in/out | common enable from the head's `core_en`, passes straight through: every board's CONVST timer starts in the same cycle |
| `rdy_o`, `all_rdy_i` | out/in | open drain on the boards = wired-AND; high when every board's Stage 3 is done |
| `up_dat_i/up_vld_i` → `up_dat_o/up_vld_o` | in/out | partial sums, 3 × 32 bits, LSB first, toward the head |

**One stride.** Each board runs Stages 1–3 on its own channels (a board with fewer channels
reads its pod sooner and finishes sooner). When `all_rdy` rises, `rdy` clears on every board in
the same cycle, the tail starts sending, every board adds its own bit (one full adder + carry
flip-flop) and passes the sum on one clock later, and the head compares each total as it arrives
and decides after the last bit (`dec_valid` → `emg_core_ctrl` → `class_valid`, first decision
masked as before). Configuration addresses as on `main`, plus `0x7F2` = K−1 (default 3).

**Test on real FORS-EMG data** (`tb/`):

```bash
python3 tb/mb_prepare.py     # from the repo root, once: writes tb/mb_data/ (already committed)
cd RTL
python3 tb/run_mb.py 4,4     # two boards; also 8 | 3,3,2 | 2,6 | 5,3 ...
python3 tb/plot_mb.py waveform/mb_4_4.vcd ../docs/waveforms
surfer waveform/mb_4_4.vcd   # or gtkwave: open the chain signals interactively
```

- `mb_prepare.py` trains the chip LDA on every subject except 6 (trials 1–4), builds W' and b',
  and takes **subject 6, trial 1**: the most active 1960 samples (2 s) of each of the 4 gestures,
  joined into one 7840-sample stream (`mb_data/samples.txt` for `$readmemh`, `samples.csv`
  readable). It runs the golden model on that stream and writes the expected decision per stride.
- `tb_mb.v` instantiates 1–3 boards side by side with one `adc_pod_model` each, shares the clock,
  CONVST-paced sample stream and cfg bus, wires the chain and prints every decision.
  `run_mb.py` slices the model per board, builds and runs it (about 40 s), and compares.
- Result: **79 of 79 decisions bit-exact** with the golden model for the splits 8, 4+4, 2+6,
  6+2, 5+3 and 3+3+2; 72 of 76 single-gesture windows correct (the 4 errors are the model's, the
  single-chip LDA makes the same decisions).
- The VCD (`waveform/mb_<split>.vcd`, about 1 MB, git-ignored) holds only the chain signals,
  each board's Stage 3 and partial sums, and the decision.

The cocotb test `tb/test_emg_top.py` is for the two-clock chip on `main` and does not run here.

# RTL on `main` — INT8 LDA gesture-classification core

Fixed-function digital core that turns 8–24 channels of 8-bit EMG samples into one gesture
decision every ~99.5 ms. Plain Verilog-2005, two clocks, no multiplier except the one 8 × 8 MAC.
Verified bit-exactly against `tb/lda_golden_model.py` by a chip-level cocotb testbench.

## System

```
 ADC pods (1-3 × 8 ch, daisy chain)
        │  SPI
 ┌──────▼───────── clk_spi 8 MHz ───┐   ┌─────────────────────── clk_sys 1 MHz ───────────────────────────┐
 │ emg_spi_adc  CONVST timer, burst │──►│ emg_feat_accum (Stage 1) ─► emg_shift_norm (Stage 2) ─► x_q RAM │
 └───────────────────┬──────────────┘   │   features, ping-pong banks     (x − μ) >> k, clip         │    │
                     │ frame_tog        │                                                             ▼    │
                cdc_toggle_sync ───────►│ emg_core_ctrl ◄──────────────────────────── emg_lda_mac (Stage 3) │
                                        │   sequencing, first-decision mask            6 classes, argmax    │
                                        │ emg_cfg_regs  μ, k, W, b, cfg_n_ch, core_en ◄── cfg port (host)   │
                                        └──────────────────────────────────────────────► class_valid/out   │
```

| | |
|---|---|
| Sample rate | 985 Hz (CONVST every 8120 `clk_spi` cycles) |
| Channels C | 1–24, register `cfg_n_ch` (8 per pod) |
| Features | 5 per channel, D = 5C: SABS, WL, ZC, SSC, POW (Mitchell square) |
| Window / stride | 196 / 98 samples (two ping-pong banks of 98) |
| Classifier | INT8 LDA, 6 class slots (unused classes: bias −2³¹), INT32 accumulator |
| Decision | `class_valid` pulse + `class_out[2:0]` every 98 samples; first decision after reset masked |

## Modules

| File | Stage | What it does |
|---|---|---|
| `emg_mitchell_sq.v` | 1 | Mitchell-1 square of \|x\|: leading-one detector, two shifts, one subtract |
| `emg_feat_alu.v` | 1 | Per-sample features of one channel (combinational): \|x\|, \|Δx\|, zero crossing, slope sign change, POW |
| `emg_tp_ram.v` | — | Generic 1-write / 1-read synchronous RAM (1-cycle read). Parameters `W`, `DEPTH`, `AW` |
| `emg_feat_accum.v` | 1 | Per frame: C channels through a read/write pipeline; history RAM (24 × 10) and two 24 × 64 feature banks; `stride_done` every 98th sample |
| `emg_shift_norm.v` | 2 | For f = 5·ch + t: bank0 + bank1 field, `clip(((raw − μ_f) + 2^(k_f−1)) >>> k_f)` → INT8 `x_q` |
| `emg_lda_mac.v` | 3 | One signed 8 × 8 multiplier: per class, bias load + D MACs; streaming argmax (strict >) |
| `emg_core_ctrl.v` | — | stride_done → Stage 2; Stage 2 done → bank swap + Stage 3; masks the first decision (`primed`) |
| `emg_cfg_regs.v` | — | Writable constants and settings (address map below) |
| `emg_spi_adc.v` | — | CONVST timer, DRDY_N synchroniser, SPI mode-0 burst of 8·C bits, publishes `rx_flat` + `frame_tog` |
| `cdc_toggle_sync.v` | — | Frame event clk_spi → clk_sys (2-FF toggle synchroniser + edge) |
| `rst_sync.v` | — | Asynchronous-assert, synchronous-release reset, one per clock domain |
| `emg_top.v` | — | Chip top: the instances above + the 120 × 8 x_q RAM |

Bank word (64 bits per channel per bank): `[13:0]` SABS, `[28:14]` WL, `[35:29]` ZC, `[42:36]` SSC, `[63:43]` POW.
Bit widths are derived in the file headers (e.g. Stage 2 needs a 24-bit signed datapath).

## Chip pins (`emg_top`)

| Pin | Dir | Width | |
|---|---|---|---|
| `clk_sys`, `clk_spi`, `arst_n` | in | 1 | 1 MHz core clock, 8 MHz ADC-interface clock, reset |
| `adc_convst`, `adc_cs_n`, `adc_sclk` | out | 1 | to all ADC pods |
| `adc_drdy_n`, `adc_miso` | in | 1 | from the pod chain |
| `cfg_we`, `cfg_addr`, `cfg_wdata` | in | 1, 12, 32 | configuration write port |
| `class_valid`, `class_out` | out | 1, 3 | decision |
| `primed`, `overrun` | out | 1 | status |

## Configuration map (`emg_cfg_regs`)

Write everything while `core_en = 0`, then set `core_en`.

| Address | Content |
|---|---|
| `0x000`–`0x077` | `{μ_f[21:0], k_f[4:0]}` for feature f = 0 … 119 |
| `0x100`–`0x3CF` | `W[cls][f]` (INT8) at `0x100 + cls·D + f` |
| `0x400`–`0x405` | `b[cls]` (INT32) |
| `0x7F0` | `cfg_n_ch` (reset 8) |
| `0x7F1` | `core_en` (reset 0) |

μ, k, W and b come from `lda_golden_model.calibrate_norm` and `fit_int8_lda`
(trained on bit-exact features); channel order is the order in which bytes leave the pod chain.

## Timing (clk_sys = 1 MHz, 1015 cycles per sample)

| Stage | Cycles |
|---|---|
| Stage 1, every sample | C |
| Stage 2, after every 98th sample | D + 2 |
| Stage 3 | 6(D + 1) + 2 |
| Measured Stage 2 + 3, C = 8 / 12 / 16 / 20 / 24 | 290 / 430 / 570 / 710 / 850 |

## Verification

```bash
cd RTL
python tb/test_emg_top.py
```

- Set `CHANNELS` (8, 12, 16, 20, 24) near the top of `tb/test_emg_top.py`.
- The test trains a model with the golden model, loads it through the config port, drives an ADC
  daisy-chain model with a gesture stream, and checks at every stride, in order:
  the D `x_q` writes (Stages 1–2), the 6 class scores (Stage 3) and `class_out`; then that the first
  decision is masked, `primed = 1` and `overrun = 0`. A failure names the first stage that disagrees.
- Run time about 1.5 min (C = 8) to 3 min (C = 24). A VCD of every signal is written to `waveform/`
  (git-ignored; large). `python tb/plot_waveforms.py` (run from `RTL/` after a C = 8 run) draws
  GTKWave-style pictures of the main events into `docs/waveforms/`.
- Passed for C = 8, 12, 16, 20 and 24. During development, the feature ALU was also checked exhaustively
  (all 196,608 input combinations) and each stage had its own unit testbench.

`tb/adc_pod_model.v` is a stand-alone behavioural ADC pod (SPI mode 0, daisy chain, `x` windows
around output changes) for plain-Verilog testbenches; it expects delays `tb.t_conv`, `tb.t_h`,
`tb.t_d`, `tb.t_en` in a top module named `tb`.

## Requirements

Icarus Verilog ≥ 11, Python 3.10+, `pip install cocotb numpy scikit-learn`. Yosys lints the whole
design (`hierarchy -check -top emg_top; proc; check -assert`) without errors.
