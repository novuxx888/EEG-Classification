# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Results

### Easier Synthetic Data
| Classifier | Accuracy |
|------------|----------|
| SVM-RBF | **88%** ⭐ |
| RandomForest | 88% |
| EEGNet | 86% |
| GradientBoosting | 86% |

### Harder Realistic Data
| Classifier | Accuracy |
|------------|----------|
| RandomForest | **74.4%** |
| ExtraTrees | 73.3% |
| GradientBoosting | 72.2% |

**Cross-validation:** 69-87% depending on data difficulty

## Methods Tried

- CSP (Common Spatial Patterns) features
- Frequency band features (alpha, beta, theta)
- Wavelet features
- Time domain features
- Multiple classifiers: LR, LDA, SVM, RF, GB, XGBoost, MLP, EEGNet

## Files

- `motor_imagery_final.py` - Best results on standard data (88%)
- `motor_imagery_v6_record.py` - Harder realistic data (74%)

## Running

```bash
python motor_imagery_final.py
python motor_imagery_v6_record.py
```

## Next Goals

- Try real motor imagery datasets
- Add more sophisticated EEG features
- Implement deep learning architectures
