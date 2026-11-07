# Results

Clips: 715 train sentences, 83 validation sentences, 200 test sentences.
Detector parameters: 91249. Epochs run: 120, best epoch 93, train time 179 s.

EER and ROC AUC (real vs fake) with 95% bootstrap intervals over clips.

| model | condition | EER | AUC |
|---|---|---|---|
| baseline | seen_clean | 0.040 [0.020, 0.060] | 0.996 [0.992, 0.998] |
| baseline | unseen_clean | 0.080 [0.055, 0.115] | 0.949 [0.925, 0.971] |
| baseline | seen_channel | 0.030 [0.010, 0.050] | 0.998 [0.995, 0.999] |
| baseline | unseen_channel | 0.085 [0.060, 0.115] | 0.960 [0.941, 0.978] |
| detector | seen_clean | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| detector | unseen_clean | 0.185 [0.150, 0.230] | 0.909 [0.881, 0.935] |
| detector | seen_channel | 0.065 [0.040, 0.090] | 0.987 [0.979, 0.993] |
| detector | unseen_channel | 0.120 [0.085, 0.160] | 0.945 [0.920, 0.966] |

Threshold set at the validation EER point (clean validation clips, MMS fakes). Error rates on test clips at that threshold:

| model | threshold | condition | false-accept rate (fake passed as real) | false-reject rate (real flagged) |
|---|---|---|---|---|
| baseline | 0.057 | seen_clean | 0.045 | 0.030 |
| baseline | 0.057 | unseen_clean | 0.320 | 0.030 |
| baseline | 0.057 | seen_channel | 1.000 | 0.000 |
| baseline | 0.057 | unseen_channel | 0.975 | 0.000 |
| detector | 11.640 | seen_clean | 0.020 | 0.000 |
| detector | 11.640 | unseen_clean | 0.940 | 0.000 |
| detector | 11.640 | seen_channel | 0.370 | 0.000 |
| detector | 11.640 | unseen_channel | 0.885 | 0.000 |

Baseline leak check (seen_clean AUC >= 0.8): LEAKS.
