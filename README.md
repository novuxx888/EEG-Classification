# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Results (on harder synthetic data)

| Classifier | Accuracy |
|------------|----------|
| RandomForest | **74.4%** |
| ExtraTrees | 73.3% |
| GradientBoosting | 72.2% |
| XGBoost | 70.0% |

**Cross-validation:** 69.3% ± 3.6%

Note: Previous 88% was on easier synthetic data. Later experiments made the data more realistic/harder.

## Methods Tried

- CSP (Common Spatial Patterns) features
- Frequency band features (alpha, beta, theta)
- Time domain features
- Multiple classifiers: LR, LDA, SVM, RF, GB, XGBoost, MLP, EEGNet
- Wavelet features

## Files

- `motor_imagery_final.py` - Best results on easier data (88%)
- `motor_imagery_v6_record.py` - Harder data (74%)

## Running

```bash
python motor_imagery_final.py   # Easier data
python motor_imagery_v6_record.py  # Harder data
```

## Next Goals

- Try real motor imagery datasets
- Add more sophisticated EEG features
- Implement deep learning architectures
