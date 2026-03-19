#!/usr/bin/env python3
"""
EEG Motor Imagery - Version 16 (Hyper-Optimized)
Building on best_combo with hyperparameter tuning and augmented data
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              VotingClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - VERSION 16 (Hyper-Optimized)")
print("="*60)

# ============================================================================
# 1. CREATE BALANCED SYNTHETIC DATA WITH AUGMENTATION
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

def create_trial(label, fs=128):
    """Create a single trial with realistic EEG"""
    t = np.arange(0, 4, 1/fs)
    n_channels = 8
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
        # Noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        # Artifacts
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # 60% show effect, 14% suppression
        show_effect = np.random.rand() < 0.60
        suppression = 0.86 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.86 + np.random.uniform(-0.1, 0.1)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.86 + np.random.uniform(-0.1, 0.1)) * suppression
        
        signals.append(base)
    
    return np.array(signals)

# Create more trials with augmentation
n_trials = 600  # More than before
X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    X.append(create_trial(label))
    y.append(label)

X = np.array(X)
y = np.array(y)

# Data augmentation: add noise-perturbed versions
print("    Augmenting data...")
X_aug = []
y_aug = []
for i in range(len(X)):
    # Add slight noise perturbation
    noise_level = 0.05
    X_noisy = X[i] + np.random.randn(*X[i].shape) * noise_level * np.std(X[i])
    X_aug.append(X_noisy)
    y_aug.append(y[i])

X = np.vstack([X, np.array(X_aug)])
y = np.concatenate([y, np.array(y_aug)])

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    (with augmentation: {len(X)//2} original + {len(X)//2} augmented)")

# ============================================================================
# 2. CSP + FBCSP FEATURES
# ============================================================================
print("\n[2] Computing CSP + FBCSP features...")

def compute_csp_for_band(X, y, fs, n_components=3, band=(8, 13)):
    """Enhanced CSP for a specific frequency band"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            cov = np.cov(trial)
            class_cov += cov / (np.trace(cov) + 1e-10)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    try:
        reg = 1e-6
        A = covs[0] + reg * np.eye(n_channels)
        B = covs[1] + reg * np.eye(n_channels)
        C = A + B + 1e-10*np.eye(n_channels)
        
        eigenvalues, eigenvectors = eigh(np.linalg.inv(C) @ A)
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, idx]
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            trial_feat = np.concatenate([
                np.log(var[:n_components] / (var[n_components:] + 1e-10)),
                var
            ])
            features.append(trial_feat)
        
        return np.array(features)
    except:
        return np.zeros((len(X), n_components * 2))

# 5-band FBCSP
bands = [
    (4, 8),    # Theta
    (8, 13),   # Mu
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (6, 12),   # Low mu
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp_for_band(X, y, 128, n_components=3, band=band)
    fbcsp_features.append(csp_feat)

fbcsp_features = np.hstack(fbcsp_features)
print(f"    FBCSP features: {fbcsp_features.shape}")

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Extracting additional features...")

# Frequency band features
def get_band_power(ch, fs, band):
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ch)
    return np.mean(filtered**2)

freq_features = []
for trial in X:
    trial_feats = []
    for ch in trial:
        # Multiple band powers
        trial_feats.append(get_band_power(ch, 128, (8, 13)))  # Alpha
        trial_feats.append(get_band_power(ch, 128, (13, 30)))  # Beta
        trial_feats.append(get_band_power(ch, 128, (4, 8)))    # Theta
        trial_feats.append(get_band_power(ch, 128, (1, 4)))    # Delta
        
        # Ratios
        alpha = trial_feats[-4]
        beta = trial_feats[-3]
        trial_feats.append(alpha / (beta + 1e-10))
        
        # Hemisphere asymmetry
        trial_feats.append(np.mean(ch))
        trial_feats.append(np.std(ch))
        trial_feats.append(np.max(ch) - np.min(ch))
    
    freq_features.append(trial_feats)

freq_features = np.array(freq_features)
print(f"    Frequency features: {freq_features.shape}")

# Hemisphere asymmetry features
def asymmetry_features(X):
    features = []
    for trial in X:
        left = trial[:4]
        right = trial[4:]
        
        left_power = np.mean([np.mean(ch**2) for ch in left])
        right_power = np.mean([np.mean(ch**2) for ch in right])
        
        asym = (right_power - left_power) / (right_power + left_power + 1e-10)
        
        # Individual channel powers
        powers = [np.mean(ch**2) for ch in trial]
        
        features.append([asym, left_power, right_power] + powers)
    
    return np.array(features)

asym_features = asymmetry_features(X)
print(f"    Asymmetry features: {asym_features.shape}")

# Combine all features
X_features = np.hstack([fbcsp_features, freq_features, asym_features])
print(f"    Combined features: {X_features.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try multiple classifiers with different hyperparameters
results = {}

# ExtraTrees - more trees
print("    Training ExtraTrees...")
et = ExtraTreesClassifier(n_estimators=1000, max_depth=25, min_samples_split=2, 
                          random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
results['ExtraTrees'] = accuracy_score(y_test, y_pred)

# GradientBoosting - more trees
print("    Training GradientBoosting...")
gb = GradientBoostingClassifier(n_estimators=400, max_depth=7, learning_rate=0.08, 
                                 subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
results['GradientBoosting'] = accuracy_score(y_test, y_pred)

# RandomForest - optimized
print("    Training RandomForest...")
rf = RandomForestClassifier(n_estimators=1000, max_depth=25, min_samples_split=2,
                           min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results['RandomForest'] = accuracy_score(y_test, y_pred)

# HistGradientBoosting (faster, often better)
print("    Training HistGradientBoosting...")
hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.08,
                                     random_state=42)
hgb.fit(X_train, y_train)
y_pred = hgb.predict(X_test)
results['HistGradientBoosting'] = accuracy_score(y_test, y_pred)

# MLP
print("    Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1500, 
                    early_stopping=True, learning_rate='adaptive', random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
results['MLP'] = accuracy_score(y_test, y_pred)

# Ensemble
print("    Training Ensemble...")
ensemble = VotingClassifier(
    estimators=[('et', et), ('gb', gb), ('rf', rf), ('hgb', hgb)],
    voting='soft', n_jobs=-1
)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
results['Ensemble'] = accuracy_score(y_test, y_pred)

# SVM
print("    Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
results['SVM-RBF'] = accuracy_score(y_test, y_pred)

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[5] RESULTS - AUGMENTED DATA (600 + 600 trials)")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_results:
    print(f"    {name}: {acc:.1%}")

best = max(results.values())
print(f"\n    BEST: {best:.1%}")

# Cross-validation on best models
print("\n[6] Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('ET', et), ('RF', rf), ('GB', gb), ('HGB', hgb)]:
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    {name}: {scores.mean():.1%} ± {scores.std():.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY - v16 with data augmentation")
print("="*60)
print("""
Key improvements:
- 600 original + 600 augmented = 1200 total trials
- Noise perturbation augmentation
- More trees (1000) in ensemble models
- HistGradientBoosting classifier
- Larger MLP (512-256-128)
- 5-band FBCSP features
- Total features: {}
""".format(X_features.shape[1]))

# Compare to previous best
prev_best = 0.825
if best > prev_best:
    print(f"✅ IMPROVED: {best:.1%} > {prev_best:.1%}")
else:
    print(f"⚠️ No improvement: {best:.1%} vs {prev_best:.1%}")
