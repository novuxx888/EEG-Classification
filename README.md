# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| ExtraTrees | **77%** |
| SVM-RBF | 75% |
| GradientBoosting | 75% |
| RandomForest | 73% |
| XGBoost | 73% |
| Ensemble | 73% |
| LogisticRegression | 67% |
| EEGNet | 65% |
| LDA | 58% |

**Cross-validation (5-fold):** RF: 83.3% ± 3.0%, SVM: 82.3% ± 2.0%, Ensemble: 81.3% ± 2.7%

## Latest Scripts

- `motor_imagery_final_v3.py` - **Best balanced version** (77% accuracy)
- `motor_imagery_final_v2.py` - Hardest version (65% accuracy)
- `motor_imagery_optimized.py` - Previous optimized version

## Methods Implemented

### Features
- **CSP (Common Spatial Patterns)** - mu band (8-13 Hz) + beta band (13-30 Hz)
- Frequency band features (alpha, beta, theta, delta powers)
- Hemisphere asymmetry features
- Time domain features (mean, std, max, IQR)

### Classifiers
- SVM-RBF (tuned C, gamma)
- RandomForest, ExtraTrees (400 trees)
- XGBoost (with libomp)
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
# Best balanced version
python3 motor_imagery_final_v3.py

# Hardest version
python3 motor_imagery_final_v2.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Implement filter bank CSP (FBCSP)
- Increase training data for EEGNet
- Try transfer learning from pre-trained EEG models
