# Results

- **Model**: ASLMLP (63 -> 128 -> 64 -> num_classes, BatchNorm + Dropout(0.3), Adam + ReduceLROnPlateau)
- **Framework**: PyTorch
- **Feature representation**: MediaPipe HandLandmarker 21 landmarks x (x,y,z), wrist-centered, scaled by wrist-to-middle-MCP distance (63-dim vector)
- **Source dataset**: Marxulia/asl_sign_languages_alphabets_v03
- **Classes (26)**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Landmark samples**: 8399 total (train 5879 / val 1260 / test 1260, 70/15/15 stratified split)
- **Hand-detection rate during extraction**: 8399/10873 (77.2%)
- **Training time**: 25.2s on CPU over 60 epochs (early stopping on val loss)

## Test set metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9183 |
| Macro precision | 0.9180 |
| Macro recall | 0.9164 |
| Macro F1 | 0.9160 |
| Test samples | 1260 |

## Per-class report (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A | 0.981 | 1.000 | 0.990 | 52 |
| B | 0.926 | 1.000 | 0.962 | 50 |
| C | 0.979 | 0.958 | 0.968 | 48 |
| D | 1.000 | 0.917 | 0.957 | 48 |
| E | 0.918 | 0.900 | 0.909 | 50 |
| F | 0.961 | 0.980 | 0.970 | 50 |
| G | 0.855 | 0.959 | 0.904 | 49 |
| H | 0.923 | 0.960 | 0.941 | 50 |
| I | 0.911 | 0.872 | 0.891 | 47 |
| J | 0.976 | 0.837 | 0.901 | 49 |
| K | 0.917 | 0.880 | 0.898 | 50 |
| L | 0.981 | 1.000 | 0.991 | 53 |
| M | 0.875 | 0.961 | 0.916 | 51 |
| N | 0.935 | 0.878 | 0.905 | 49 |
| O | 0.875 | 1.000 | 0.933 | 42 |
| P | 0.841 | 0.787 | 0.813 | 47 |
| Q | 0.861 | 0.816 | 0.838 | 38 |
| R | 0.837 | 0.872 | 0.854 | 47 |
| S | 0.870 | 0.959 | 0.913 | 49 |
| T | 0.958 | 0.902 | 0.929 | 51 |
| U | 0.833 | 0.816 | 0.825 | 49 |
| V | 0.957 | 0.917 | 0.936 | 48 |
| W | 0.978 | 0.957 | 0.968 | 47 |
| X | 0.898 | 0.936 | 0.917 | 47 |
| Y | 0.946 | 1.000 | 0.972 | 53 |
| Z | 0.875 | 0.761 | 0.814 | 46 |

Confusion matrix image: `results_confusion_matrix.png`. Full raw numbers: `results.json`.
