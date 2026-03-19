# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| **SVM-RBF (FBCSP)** | **81%** ← NEW! |
| RandomForest | 80% |
| ExtraTrees | 80% |
| GradientBoosting | 80% |
| XGBoost | 80% |
| Ensemble | 80% |
| LogisticRegression | 79% |
| LightGBM | 79% |
| EEGNet | 76% |
| LDA | 66% |

**Cross-validation (5-fold):** RF: 82.3% ± 5.2%, SVM: 81.4% ± 5.4%, Ensemble: 82.3% ± 5.3%, LightGBM: 79.7% ± 5.5%

## Latest Scripts

- `motor_imagery_fbcsp_balanced.py` - **Best: 81% accuracy (NEW!)** ← Run this!
- `motor_imagery_fbcsp.py` - Hard version (70% accuracy)
- `motor_imagery_final_v3.py` - Previous best (77% accuracy)

## Methods Implemented

### Features
- **FBCSP (Filter Bank CSP)** - Multiple bands: mu (8-13 Hz), beta1 (13-20 Hz), beta2 (20-30 Hz)
- **CSP (Common Spatial Patterns)** - mu band (8-13 Hz) + beta band (13-30 Hz)
- Frequency band features (alpha, beta, theta, delta powers)
- Hemisphere asymmetry features
- Time domain features (mean, std, max, IQR)

### Classifiers
- SVM-RBF (tuned C, gamma) ← **Best performer!**
- RandomForest, ExtraTrees (400-500 trees)
- XGBoost
- LightGBM (NEW!)
- Gradient Boosting
- Logistic Regression, LDA
- Voting Ensemble

### Deep Learning
- EEGNet architecture (TensorFlow/Keras)
- Custom CNN with spatial filtering

## Synthetic Data

Balanced difficulty (current best):
- 300 trials, 8 channels, 3.5 seconds (128 Hz)
- Multiple EEG rhythms (alpha ~10Hz, beta ~18-24Hz, theta ~6Hz)
- Realistic noise (white noise, drift, artifacts)
- Cross-trial variability (0.4x to 1.6x amplitude)
- **65% of trials show motor imagery effect (15% suppression)**

Hard version (more challenging):
- 55% trials show effect, only 10% suppression
- More noise and variability

## Key Insights

1. **CSP features are critical** - they maximize class separability in motor imagery
2. **Single-band CSP (mu)** works better than multi-band for this data
3. **ExtraTrees/SVM-RBF** perform best on this data (75-77%)
4. **XGBoost** now works with libomp installed
5. **EEGNet** achieves 65% - needs more data or pre-trained models
6. **Cross-validation** shows 80-83% potential with proper tuning
7. **Harder synthetic data** produces lower but more realistic accuracy

## Running

```bash
# New best (81% accuracy!)
python3 motor_imagery_fbcsp_balanced.py

# Hard version (70% accuracy)
python3 motor_imagery_fbcsp.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Implement filter bank CSP (FBCSP)
- Increase training data for EEGNet
- Try transfer learning from pre-trained EEG models
