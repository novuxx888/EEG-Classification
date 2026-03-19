# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| **RandomForest** | **81.4%** ← NEW RECORD! |
| **XGBoost** | **81.4%** |
| **LightGBM** | **81.4%** |
| **EEGNet** | **81.4%** |
| **Ensemble** | **81.4%** |
| ExtraTrees | 80.0% |
| GradientBoosting | 80.0% |
| SVM-RBF | 74.3% |
| LogisticRegression | 72.9% |
| LDA | 54.3% |

**Cross-validation (5-fold):** RF: 80.3% ± 5.0%, SVM: 78.9% ± 4.5%, ET: 77.4% ± 6.2%, GB: 77.4% ± 4.3%

## Latest Scripts

- `motor_imagery_fbcsp_v4.py` - **Best: 81.4% accuracy** ← Run this!
- `motor_imagery_csp_v2.py` - Balanced-hard (78.6%)
- `motor_imagery_csp_rf_xgb.py` - Hard data version (68%)
- `motor_imagery_fbcsp_balanced.py` - Previous best (81%)

## Methods Implemented

### Features
- **FBCSP (Filter Bank CSP)** - 4 bands: theta (4-8 Hz), mu (8-13 Hz), beta1 (13-20 Hz), beta2 (20-30 Hz)
- **CSP (Common Spatial Patterns)** - mu + beta bands
- Frequency band features (alpha, beta, theta, delta powers)
- Hemisphere asymmetry features
- Spatial pattern features
- Time domain features (mean, std, max, IQR)

### Classifiers
- RandomForest (500 trees, max_depth=15) ← **NEW Best!**
- XGBoost, LightGBM (tuned hyperparameters)
- SVM-RBF (tuned C, gamma)
- ExtraTrees, Gradient Boosting
- Logistic Regression, LDA
- Voting Ensemble

### Deep Learning
- EEGNet architecture (TensorFlow/Keras) - **Now achieves 81.4%!**
- Custom CNN with spatial filtering

## Synthetic Data

Balanced difficulty (current best):
- 350 trials, 8 channels, 3.5 seconds (128 Hz)
- Multiple EEG rhythms (alpha ~10Hz, beta ~18-22Hz, theta ~6Hz)
- Realistic noise (white noise, drift, artifacts)
- Cross-trial variability (0.4x to 1.6x amplitude)
- **60% of trials show motor imagery effect (14% suppression)**

Hard version:
- 50% trials show effect, only 12% suppression
- More noise and variability (~68% accuracy)

## Key Insights

1. **FBCSP with 4 bands is powerful** - theta, mu, beta1, beta2
2. **RandomForest/XGBoost/LightGBM tied at 81.4%** - all perform equally well
3. **EEGNet improved significantly** - now achieves 81.4% with deeper architecture
4. **Ensemble matches best individual** - but no improvement over single models
5. **Cross-validation shows ~77-80%** potential with variance
6. **SVM-RBF dropped** - tree-based methods now outperform
7. **Balanced data (60% effect)** is the sweet spot for this synthetic dataset

## Running

```bash
# New best (81.4% accuracy!)
python3 motor_imagery_fbcsp_v4.py

# Balanced version (78.6%)
python3 motor_imagery_csp_v2.py

# Hard version (68%)
python3 motor_imagery_csp_rf_xgb.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Add more channels for better spatial resolution
- Try other deep learning architectures (ShallowConvNet, DeepConvNet)
- Experiment with transfer learning
