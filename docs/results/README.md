# Results

Raw printouts of the scripts behind the numbers in the paper and the slides.

| File | Produced by | Data |
|---|---|---|
| `merge_vs_drop.txt` | `analysis/K_4_LDA_Merge_vs_Drop.py` | 19 subjects |
| `cnn_loso_19_subjects.txt` | `analysis/K_4_CNN_LOSO.py` (≤ 150 epochs, GPU) | 19 subjects |
| `lda_calibration_18_subjects.txt` | `analysis/K_4_LDA_Calib.py` | 18 subjects (repository copy of the dataset) |
| `rtl_verify_all.txt` | `RTL/tb/verify_all.py` | chain-link unit test, mutation test, whole chip on subject 6, trial 1 |
| `CNN_checkpoint.pdf` | CNN branch report: CNN vs LDA before and after calibration, arm-position check | 19 subjects |
| `../report/outputs/script_outputs.txt` | golden model, integer feature variants, feature selection | 18 subjects |

The CNN checkpoint uses its own setting (12 gestures mapped to 6 groups, 150 ms windows with a 50 ms stride,
held windows only); compare its numbers only with each other.
