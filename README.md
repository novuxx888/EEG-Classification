# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| SVM-RBF | **88%** |
| RandomForest | 88% |
| GradientBoosting | 86% |
| EEGNet | 86% |
| LogisticRegression | 84% |

**Cross-validation:** 87.2% ± 8.2%

## Methods Tried

- CSP (Common Spatial Patterns) features
- Frequency band features (alpha, beta, theta)
- Time domain features
- Multiple classifiers: LR, LDA, SVM, RF, GB, XGBoost, EEGNet

## Running

```bash
python motor_imagery_final.py
```

## Next Goals

- Try real motor imagery datasets
- Add more sophisticated EEG features
- Implement deep learning architectures
