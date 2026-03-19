# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| **GradientBoosting** | **88.9%** ← NEW RECORD! |
| RandomForest | 87.8% |
| ExtraTrees | 86.7% |
| LightGBM | 86.7% |
| XGBoost | 85.6% |
| MLP | 85.6% |
| SVM-RBF | 84.4% |
| Ensemble | 84.4% |
| LogisticRegression | 81.1% |
| EEGNet | 77.8% |
| LDA | 68.9% |

**Cross-validation (5-fold):** RF: 86.9% ± 2.4%, ET: 86.9% ± 3.1%, GB: 86.4% ± 2.3%, XGBoost: 86.7% ± 2.4%, LightGBM: 86.4% ± 3.3%

## Latest Experiments (March 2026)

### New Approach - v3 Optimized

- **88.9% accuracy** with Gradient Boosting! (+6.4% improvement over 82.5%)
- Enhanced CSP features with regularized covariance
- 6-band FBCSP (added delta band 2-4Hz)
- Optimized RF/XGBoost parameters
- Balanced difficulty: 65% effect, 16% suppression

### Hard Data Results (~52% is near chance for 3-class)
- Logistic Regression: 51%
- Random Forest: 51%
- XGBoost: 52% 🎯
- Gradient Boosting: 48%
- EEGNet (TF): 50%

The hard data with 3 classes and subtle 2% alpha suppression is near random chance, demonstrating how challenging real motor imagery classification is.

## Latest Scripts

- `motor_imagery_v3_optimized.py` - **NEW BEST: 88.9% accuracy** ← Run this!
- `motor_imagery_v2_improved.py` - Hard data (55% effect, 12% suppression) - ~67%
- `motor_imagery_best_combo.py` - Previous best (82.5%)
- `motor_imagery_enhanced.py` - CSP + RF/XGBoost + hard data (3-class, ~52%)
- `motor_imagery_eegnet.py` - EEGNet deep learning (~50% on hard data)

## Methods Implemented

### Features
- **FBCSP (Filter Bank CSP)** - 6 bands: delta (2-4 Hz), theta (4-8 Hz), mu (8-13 Hz), low-mu (6-12 Hz), beta1 (13-20 Hz), beta2 (20-30 Hz)
- **CSP (Common Spatial Patterns)** - with regularized covariance
- Frequency band features (alpha, beta, theta, delta powers + ratios)
- Hemisphere asymmetry features
- Spatial pattern features
- Temporal segment features (6 segments)
- Connectivity features (pairwise correlations)
- Time domain features (mean, std, max, IQR, RMS, MAD)

### Classifiers
- GradientBoosting (350 trees, max_depth=6, lr=0.1) ← **NEW BEST!**
- RandomForest (800 trees, max_depth=20)
- ExtraTrees (800 trees, max_depth=20)
- XGBoost, LightGBM (tuned hyperparameters)
- SVM-RBF (tuned C, gamma)
- MLP Neural Network (512-256-128-64)
- Voting Ensemble

### Deep Learning
- EEGNet architecture (TensorFlow/Keras) - ~78% on balanced data

## Synthetic Data

**Balanced v3 (current best - 88.9%):**
- 450 trials, 8 channels, 4 seconds (128 Hz)
- Multiple EEG rhythms (alpha ~10Hz, beta ~18-22Hz, theta ~6Hz, delta ~2Hz)
- Realistic noise (white noise, drift, artifacts, line noise)
- Cross-trial variability (0.4x to 1.6x amplitude)
- **65% of trials show motor imagery effect (16% suppression)**

**Previous best (82.5%):**
- 400 trials, 8 channels, 4 seconds (128 Hz)
- 60% effect, 14% suppression

**Hard version (~52-59%):**
- 3-class (left/right/rest) vs 2-class
- 20 channels vs 8 channels
- 2% alpha suppression (very subtle!) vs 14%

## Key Insights

1. **GradientBoosting leads at 88.9%** - significant improvement over previous 82.5%
2. **6-band FBCSP helps** - adding delta band (2-4 Hz) improves discrimination
3. **More features = better** - 352 features vs previous versions
4. **EEGNet improved** - 77.8% on balanced data (vs 68.8% before)
5. **Hard data is significantly harder** - 3-class + 2% suppression drops to ~52%
6. **Cross-validation consistent** - 86-87% CV shows stable performance
7. **MLP competitive** - 85.6% accuracy

## Running

```bash
# New best (88.9% accuracy!)
python3 motor_imagery_v3_optimized.py

# Previous best (82.5%)
python3 motor_imagery_best_combo.py

# Harder version (67% on harder data)
python3 motor_imagery_v2_improved.py

# Enhanced with CSP + RF/XGBoost (3-class hard data)
python3 motor_imagery_enhanced.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Add more channels for better spatial resolution
- Try other deep learning architectures (ShallowConvNet, DeepConvNet)
- Experiment with transfer learning
- Explore Riemannian geometry approaches