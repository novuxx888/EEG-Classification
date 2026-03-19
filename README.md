# EEG Motor Imagery Classification 🧠

Practice project for classifying EEG thought patterns (motor imagery).

## Current Best Results

| Classifier | Accuracy |
|------------|----------|
| **ExtraTrees** | **82.5%** ← NEW RECORD! |
| **GradientBoosting** | **82.5%** |
| MLP | 80.0% |
| RandomForest | 78.8% |
| LightGBM | 78.8% |
| Ensemble | 78.8% |
| XGBoost | 76.2% |
| SVM-RBF | 75.0% |
| LogisticRegression | 68.8% |
| EEGNet | 68.8% |
| LDA | 60.0% |

**Cross-validation (5-fold):** RF: 79.2% ± 2.3%, ET: 77.8% ± 2.0%, XGBoost: 77.2% ± 0.9%

## Latest Scripts

- `motor_imagery_best_combo.py` - **Best: 82.5% accuracy** ← Run this!
- `motor_imagery_harder_v2.py` - Hard data (40% effect, 10% suppression) - ~59%
- `motor_imagery_fbcsp_v4.py` - Previous best (81.4%)

## Methods Implemented

### Features
- **FBCSP (Filter Bank CSP)** - 5 bands: theta (4-8 Hz), mu (8-13 Hz), low-mu (6-12 Hz), beta1 (13-20 Hz), beta2 (20-30 Hz)
- **CSP (Common Spatial Patterns)** - with regularization
- Frequency band features (alpha, beta, theta, delta powers + ratios)
- Hemisphere asymmetry features
- Spatial pattern features
- Temporal segment features (5 segments)
- Time domain features (mean, std, max, IQR, RMS)

### Classifiers
- ExtraTrees (700 trees, max_depth=20) ← **NEW Best!**
- GradientBoosting (300 trees, max_depth=6)
- RandomForest (700 trees, max_depth=20)
- XGBoost, LightGBM (tuned hyperparameters)
- SVM-RBF (tuned C, gamma)
- MLP Neural Network
- Voting Ensemble

### Deep Learning
- EEGNet architecture (TensorFlow/Keras)
- Enhanced architecture with deeper networks

## Synthetic Data

**Balanced (current best - 82.5%):**
- 400 trials, 8 channels, 4 seconds (128 Hz)
- Multiple EEG rhythms (alpha ~10Hz, beta ~18-22Hz, theta ~6Hz)
- Realistic noise (white noise, drift, artifacts)
- Cross-trial variability (0.4x to 1.6x amplitude)
- **60% of trials show motor imagery effect (14% suppression)**

**Hard version (~59%):**
- 40% trials show effect, 10% suppression
- More noise and variability
- Even tree-based methods struggle

## Key Insights

1. **ExtraTrees/GradientBoosting lead at 82.5%** - slight edge over RF/XGBoost
2. **More features = better performance** - 256 features vs previous
3. **5-band FBCSP helps** - adding low-mu (6-12 Hz) improves discrimination
4. **EEGNet underperforms on this data** - 68.8% vs 80%+ for classical ML
5. **Hard data is significantly harder** - 40% effect / 10% suppression drops to ~59%
6. **Cross-validation consistent** - 77-79% CV shows stable performance
7. **MLP competitive** - 80% accuracy, good alternative

## Running

```bash
# New best (82.5% accuracy!)
python3 motor_imagery_best_combo.py

# Previous best (81.4%)
python3 motor_imagery_fbcsp_v4.py

# Hard version (~59%)
python3 motor_imagery_harder_v2.py
```

## Next Goals

- Try real motor imagery datasets (BCI Competition IV)
- Add more channels for better spatial resolution
- Try other deep learning architectures (ShallowConvNet, DeepConvNet)
- Experiment with transfer learning
- Explore Riemannian geometry approaches
