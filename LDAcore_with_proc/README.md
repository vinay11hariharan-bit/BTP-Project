# LDAcore_with_proc — configurable EMG feature front end for a microcontroller

Branch `processor_flexibility` (Sudhindra, 31 Oct – 1 Nov 2026).

This version of the chip **extracts features only**. The on-chip LDA (Stage 3) is removed: the chip
streams each window's 5C INT8 features to a microcontroller over SPI, and the gesture decision runs
in firmware. Any classifier (LDA, logistic regression, a small network) can then be used or updated
without new silicon. The ADC side talks to real **TI ADS1299** devices.

## What changed from `main` (in the order it was built)

| Step | Change |
|---|---|
| 1 | 1 kHz sampling; window = N banks of B samples, both registers (B 1–100, N 1–4); warm-up count W replaces the first-decision mask |
| 2 | ADS1299 interface: 24-bit codes, status words, SPI mode 1, device setup commands from the chip; pods of 8, 6 or 4 channels; normaliser μ widened to 24 bits, k signed (−32 … 31) |
| 3 | Microcontroller link `emg_spi_slave` (chip = SPI slave, ≤ 250 kHz): read frame, write setting, read setting; on-chip LDA and the parallel settings bus removed; range-checked settings with a `cfg_err` flag |
| 4 | Health flags per frame: rest (activity below T_REST), dead channel (24-bit code unchanged over the window), ADS1299 lead-off, link error, skipped frame, overrun |
| 5 | Optional on-chip input filter `emg_filter`: 20–450 Hz band-pass + 50 Hz notch, 5 fixed-point biquads on one shared 16 × 24 multiplier (480 of 1000 cycles at C = 24); coefficients writable over SPI |

## Chip pins (`emg_top`)

| Group | Pins |
|---|---|
| Clocks, reset | `clk_sys` (1 MHz), `clk_spi` (8 MHz), `arst_n` |
| ADS1299 chain | `adc_convst` (START), `adc_drdy_n`, `adc_cs_n`, `adc_sclk`, `adc_miso` (DOUT of device 0), `adc_mosi` (DIN, setup commands) |
| Microcontroller | `mcu_cs_n`, `mcu_sclk`, `mcu_mosi`, `mcu_miso`, `mcu_drdy` (new frame ready) |

## Microcontroller protocol (`emg_spi_slave`, SPI mode 0, MSB first)

| Command | Bytes from the MCU | Chip sends |
|---|---|---|
| `0x01` read frame | — | frame below |
| `0x02` write setting | 2 address + 4 value | — |
| `0x03` read setting | 2 address + 1 spare | 4 value bytes |

Frame (5C + 6 bytes): counter, flags, 3-byte dead-channel mask, 5C signed features
(f = 5·ch + type), checksum. The checksum is sent inverted if the frame changed while it was being read.

## Settings (`emg_cfg_regs`)

| Address | Setting |
|---|---|
| `0x000`–`0x077` | `{k_f[5:0], μ_f[23:0]}` per feature |
| `0x7C0`–`0x7D0` | filter coefficients (signed 16-bit, 14 fractional bits) |
| `0x7F0` | C, channels (reset 8) |
| `0x7F1` | `core_en` |
| `0x7F2` | channels per ADS1299 device: 8, 6 or 4 |
| `0x7F3` | 24-bit code → INT8 shift (filter off) or filter output shift (filter on) |
| `0x7F4` | ADS1299 command byte (core_en = 0 only; ≥ 10 µs between writes) |
| `0x7F5` / `0x7F6` / `0x7F7` | bank size B / bank count N / warm-up W |
| `0x7F8` | rest threshold T_REST |
| `0x7F9` | dead-check rule (all 24 bits or low 8 bits) |
| `0x7FA` / `0x7FB` | status (read only) / rest sum of the last frame (read only) |
| `0x7FC` | filter on/off |

Out-of-range writes to C, B or N set `cfg_err` and turn `core_en` off; enabling is refused until fixed.

## Files

| File | Purpose |
|---|---|
| `emg_top.v` | top level and wiring |
| `emg_spi_adc.v` | ADS1299 daisy-chain controller (START, DRDY, mode-1 read, command port, lead-off and header checks) |
| `emg_filter.v` | 5-section fixed-point IIR input filter |
| `emg_feat_alu.v`, `emg_mitchell_sq.v`, `emg_feat_accum.v` | Stage 1 features and N-bank accumulation |
| `emg_shift_norm.v` | Stage 2: 24-bit μ, signed k, dead mask, rest sum |
| `emg_core_ctrl.v` | sequencing, skip rule, frame header and status |
| `emg_spi_slave.v` | microcontroller link |
| `emg_cfg_regs.v` | settings |
| `emg_tp_ram.v`, `cdc_toggle_sync.v`, `rst_sync.v` | RAM, clock-domain crossing, reset synchronisers |
| `emg_lda_mac.v` | kept from `main`, not instantiated |
| `tb/golden_model.py` | one bit-exact model of chip + processor; presets `legacy` (main) and `target` (this design) |
| `tb/lda_golden_model.py` | compatibility layer; `tb/check_compat.py` proves it equals the original |
| `tb/test_emg_top.py` | chip test with a Python ADS1299 model and an MCU model (`tb/mcu_model.py`) |
| `tb/test_emg_chain.py` + `tb/tb_chain.v` + `tb/ads1299_model.v` | chip with a chain of Verilog ADS1299 models, datasheet timing checks and fault injection |
| `tb/test_filter.py`, `tb/test_shift_norm.py` | unit tests |
| `run_test_step1.sh` … `run_test_step5.sh`, `run_test.sh` | regressions; logs in `logs_step*/` |

## Running

From this folder:

```bash
bash run_test_step5.sh        # full regression for steps 1-5 (1-6 min per test)
python tb/test_emg_top.py     # one chip test; EMG_FILTER=1 turns the filter on
python tb/test_emg_chain.py --pods 3 --leadoff 4,9 --stuck 16,17   # chain with injected faults
```

`test_emg_chain.py` options cover pod counts and sizes (`--pods`, `--pod-ch`, `--channels`), device
mismatch (`--cfg-dev-ch`), lead-off, stuck and quiet channels, dropped or late DRDY, and timing
limits (`--t-d`, `--cmd-gap`). The cases `td130` and `gap3` are expected to fail: they break an
ADS1299 timing rule on purpose.

## Status

All regression runs pass except the two intentional failures. Open questions are marked `OPEN-Qn`
in `tb/golden_model.py` (for example rounding at the INT8 limits, the dead check on the first sample
after reset, and the optional ADS1299 sinc3 model); this README is where the golden model expects
them to be listed.

Before merging into `main`: this branch was committed from a fresh folder, so it deletes the Python
training scripts and `RTL/` of `main`, and it adds simulator build files (`sim_build*/`) and logs that
should be git-ignored.
