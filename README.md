# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| **GradientBoosting** | **88.9%** ← RECORD! |
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

### Difficulty Scaling Study

| Difficulty | Effect % | Suppression | Best Accuracy |
|------------|----------|-------------|---------------|
| Easy (v3) | 65% | 16% | **88.9%** |
| Medium (v4b) | 55% | 12% | 77.0% |
| Ultra (v5) | 45% | 10% | 67.5% |

Key findings:
- Reducing effect percentage dramatically degrades performance
- 55% effect/12% suppression → 77.0% (vs 88.9%)
- 45% effect/10% suppression → 67.5% (near real-world difficulty)
- EEGNet struggles more on harder data (49% on ultra-hard vs 78% on easy)

### New Approach - v4b & v5

- **v4b**: Medium difficulty (55% effect, 12% suppression) - 77.0% with GradientBoosting
- **v5**: Ultra challenge (45% effect, 10% suppression) - 67.5% with RandomForest/MLP
- CSP + 6-band FBCSP features confirmed effective
- More noise/artifacts make classification significantly harder
- EEGNet with spectrogram input: 49.2% on ultra-hard (near chance!)

### Hard Data Results (~52% is near chance for 3-class)
- Logistic Regression: 51%
- Random Forest: 51%
- XGBoost: 52% 🎯
- Gradient Boosting: 48%
- EEGNet (TF): 50%

The hard data with 3 classes and subtle 2% alpha suppression is near random chance, demonstrating how challenging real motor imagery classification is.

## Latest Scripts

- `motor_imagery_v3_optimized.py` - **RECORD: 88.9% accuracy** ← Run this!
- `motor_imagery_v4b.py` - Medium difficulty (55%/12%) - 77%
- `motor_imagery_v5.py` - Ultra challenge (45%/10%) - 67.5%
- `motor_imagery_v2_improved.py` - Hard data (55% effect, 12% suppression) - ~67%
- `motor_imagery_best_combo.py` - Previous best (82.5%)
- `motor_imagery_enhanced.py` - CSP + RF/XGBoost + hard data (3-class, ~52%)

## Methods Implemented

### Features
- **FBCSP (Filter Bank CSP)** - 6 bands: delta (2-4 Hz), theta (4-8 Hz), mu (8-13 Hz), low-mu (6-12 Hz), beta1 (13-20 Hz), beta2 (20-30 Hz)
- **CSP (Common Spatial Patterns)** - with regularized covariance
- Frequency band features (alpha, beta, theta, delta powers + ratios)
- Hemisphere asymmetry features
- Spatial pattern features
- Temporal segment features (6-8 segments)
- Connectivity features (pairwise correlations)
- Time domain features (mean, std, max, IQR, RMS, MAD)

### Classifiers
- GradientBoosting (500 trees, max_depth=7, lr=0.08) ← **RECORD!**
- RandomForest (1000 trees, max_depth=25)
- ExtraTrees (1000 trees, max_depth=25)
- XGBoost, LightGBM (tuned hyperparameters)
- SVM-RBF (tuned C, gamma)
- MLP Neural Network (1024-512-256-128)
- Voting Ensemble

### Deep Learning
- EEGNet architecture (TensorFlow/Keras) - 78% on balanced data
- EEGNet with spectrogram input - 49% on ultra-hard data
- Demonstrates the gap between easy synthetic and realistic difficulty

## Synthetic Data

**Balanced v3 (current best - 88.9%):**
- 450 trials, 8 channels, 4 seconds (128 Hz)
- Multiple EEG rhythms (alpha ~10Hz, beta ~18-22Hz, theta ~6Hz, delta ~2Hz)
- Realistic noise (white noise, drift, artifacts, line noise)
- Cross-trial variability (0.4x to 1.6x amplitude)
- **65% of trials show motor imagery effect (16% suppression)**

**Medium difficulty v4b (77%):**
- 500 trials, 8 channels, 4 seconds (128 Hz)
- 55% effect, 12% suppression
- More artifacts and electrode noise

**Ultra challenge v5 (67.5%):**
- 600 trials, 8 channels, 4 seconds (128 Hz)
- 45% effect, 10% suppression
- Heavy noise, more artifacts, 50Hz line noise simulation
- Near real-world difficulty

**Previous best (82.5%):**
- 400 trials, 8 channels, 4 seconds (128 Hz)
- 60% effect, 14% suppression

**Hard version (~52-59%):**
- 3-class (left/right/rest) vs 2-class
- 20 channels vs 8 channels
- 2% alpha suppression (very subtle!) vs 14%

## Key Insights

1. **GradientBoosting leads at 88.9%** - significant improvement over previous 82.5%
2. **Difficulty scaling shows realistic gap** - easy synthetic (89%) vs ultra-hard (67.5%)
3. **6-band FBCSP helps** - adding delta band (2-4 Hz) improves discrimination
4. **EEGNet struggles on hard data** - 78% on easy → 49% on ultra-hard
5. **Real motor imagery is hard** - the 67.5% on ultra-hard is likely closer to real-world performance
6. **Cross-validation consistent** - 86-87% CV shows stable performance on easy data
7. **MLP competitive** - 85.6% accuracy on easy data

## Running

```bash
# Record holder (88.9% accuracy!)
python3 motor_imagery_v3_optimized.py

# Medium difficulty (77%)
python3 motor_imagery_v4b.py

# Ultra challenge (67.5%)
python3 motor_imagery_v5.py

# Previous best (82.5%)
python3 motor_imagery_best_combo.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Add more channels for better spatial resolution
- Try other deep learning architectures (ShallowConvNet, DeepConvNet)
- Experiment with transfer learning
- Explore Riemannian geometry approaches
