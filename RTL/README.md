# RTL — INT8 LDA gesture-classification core

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
  (git-ignored; large).
- Passed for C = 8, 12, 16, 20 and 24. During development, the feature ALU was also checked exhaustively
  (all 196,608 input combinations) and each stage had its own unit testbench.

`tb/adc_pod_model.v` is a stand-alone behavioural ADC pod (SPI mode 0, daisy chain, `x` windows
around output changes) for plain-Verilog testbenches; it expects delays `tb.t_conv`, `tb.t_h`,
`tb.t_d`, `tb.t_en` in a top module named `tb`.

## Requirements

Icarus Verilog ≥ 11, Python 3.10+, `pip install cocotb numpy scikit-learn`. Yosys lints the whole
design (`hierarchy -check -top emg_top; proc; check -assert`) without errors.
