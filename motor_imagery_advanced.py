#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Advanced Version
With CSP features, RandomForest, XGBoost, harder synthetic data
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY CLASSIFICATION - ADVANCED")
print("="*60)

# ============================================================
# 1. CREATE HARDER SYNTHETIC DATA (More realistic & noisy)
# ============================================================
print("\n[1] Creating harder synthetic motor imagery data...")

fs = 128  # Sampling frequency
n_trials = 200  # More trials
n_channels = 4   # C3, C4, Fz, Pz

def generate_realistic_eeg(n_channels, n_samples, fs=128):
    """Generate realistic EEG background with multiple noise sources"""
    t = np.arange(0, n_samples/fs, 1/fs)
    
    # Mix of oscillators at different frequencies
    signal = np.zeros(n_samples)
    
    # Add key EEG frequencies with random phases
    for freq in [5, 7, 10, 12, 15, 20, 25]:
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(5, 15)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add pink noise (1/f)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    pink = np.random.randn(len(freqs))
    pink = pink / (freqs + 0.5)**0.7
    signal += np.fft.irfft(pink, n_samples) * 8
    
    # Add random noise bursts (eye blinks, muscle artifacts)
    for _ in range(3):
        idx = np.random.randint(0, n_samples - 50)
        burst = np.random.randn(50) * 30
        signal[idx:idx+50] += burst
    
    return signal

def generate_harder_motor_imagery_trial(label, fs=128, t_len=4):
    """
    Generate synthetic motor imagery trial that is MUCH harder to classify
    - Subtle differences between classes
    - More noise
    - Not always consistent
    """
    n_samples = int(fs * t_len)
    
    # Channel setup: C3(ch0), C4(ch1), Fz(ch2), Pz(ch3)
    X = np.zeros((n_channels, n_samples))
    
    # Generate realistic background for each channel
    for ch in range(n_channels):
        X[ch] = generate_realistic_eeg(n_channels, n_samples, fs)
    
    # MOTOR IMAGERY EFFECT - Make it SUBTLE
    # Only 60% of trials show clear effect (mimics real data)
    has_clear_effect = np.random.random() < 0.6
    
    if has_clear_effect:
        t = np.arange(0, t_len, 1/fs)
        
        # Mu rhythm (alpha, 8-13 Hz) - strongest over motor cortex
        # ERD (Event-Related Desynchronization) during motor imagery
        
        if label == 0:  # LEFT hand imagery
            # Suppress alpha over RIGHT motor cortex (C4 = ch1)
            mu = np.sin(2 * np.pi * 10 * t)
            # Only 30-50% suppression (was 70%)
            suppression = 0.5 + np.random.random() * 0.2
            mask = np.ones_like(t) - (1-suppression) * (mu**2)
            X[1] = X[1] * mask + np.random.randn(n_samples) * 5  # Add noise
            
            # Also subtle effect on C3 (contralateral - less expected but present)
            mask2 = np.ones_like(t) - 0.15 * (mu**2)
            X[0] = X[0] * mask2 + np.random.randn(n_samples) * 3
            
        else:  # RIGHT hand imagery  
            # Suppress alpha over LEFT motor cortex (C3 = ch0)
            mu = np.sin(2 * np.pi * 10 * t)
            suppression = 0.5 + np.random.random() * 0.2
            mask = np.ones_like(t) - (1-suppression) * (mu**2)
            X[0] = X[0] * mask + np.random.randn(n_samples) * 5
            
            # Subtle effect on C4
            mask2 = np.ones_like(t) - 0.15 * (mu**2)
            X[1] = X[1] * mask2 + np.random.randn(n_samples) * 3
    
    # Add channel-specific noise (different per trial)
    for ch in range(n_channels):
        X[ch] += np.random.randn(n_samples) * np.random.uniform(5, 15)
    
    # Add some trials where the effect is wrong/mislabeled (label noise)
    if np.random.random() < 0.05:  # 5% label noise
        label = 1 - label
    
    return X, label

# Generate harder dataset
X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    trial_data, label = generate_harder_motor_imagery_trial(label)
    X.append(trial_data)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Generated: {X.shape} (trials, channels, samples)")
print(f"    Labels: left={np.sum(np.array(y)==0)}, right={np.sum(np.array(y)==1)}")

# ============================================================
# 2. CSP FEATURE EXTRACTION
# ============================================================
print("\n[2] Implementing CSP feature extraction...")

def compute_covariance(X_trial):
    """Compute spatial covariance matrix"""
    X_trial = X_trial - X_trial.mean(axis=1, keepdims=True)
    N = X_trial.shape[1]
    return np.dot(X_trial, X_trial.T) / N

def compute_csp_filters(X, y, n_filters=2):
    """Compute CSP spatial filters"""
    X_class0 = X[np.array(y) == 0]
    X_class1 = X[np.array(y) == 1]
    
    cov0 = np.mean([compute_covariance(trial) for trial in X_class0], axis=0)
    cov1 = np.mean([compute_covariance(trial) for trial in X_class1], axis=0)
    
    eigenvalues, eigenvectors = eigh(cov0, cov1)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Take first and last n_filters (most discriminative)
    filters = np.hstack([eigenvectors[:, :n_filters], eigenvectors[:, -n_filters:]])
    return filters

def extract_csp_features(X, filters):
    """Extract CSP log-variance features"""
    features = []
    for trial in X:
        projected = np.dot(filters.T, trial)  # (2*n_filters, n_samples)
        trial_features = []
        for ch in range(projected.shape[0]):
            var = np.var(projected[ch])
            trial_features.append(np.log(var + 1e-10))
        features.append(trial_features)
    return np.array(features)

# Bandpass filter for CSP (mu+beta: 8-30 Hz)
def bandpass_filter_trial(X_trial, fs, low=8, high=30):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, X_trial, axis=1)

X_filtered = np.array([bandpass_filter_trial(x, fs) for x in X])
csp_filters = compute_csp_filters(X_filtered, y, n_filters=2)
X_csp = extract_csp_features(X_filtered, csp_filters)
print(f"    CSP features: {X_csp.shape}")

# ============================================================
# 3. FREQUENCY BAND FEATURES
# ============================================================
print("\n[3] Extracting frequency band features...")

def extract_band_features(X, fs=128):
    """Extract power in different frequency bands"""
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    
    features = []
    for trial in X:
        trial_features = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            for band, (lo, hi) in bands.items():
                idx = np.logical_and(freqs >= lo, freqs <= hi)
                trial_features.append(np.mean(psd[idx]))
        features.append(trial_features)
    return np.array(features)

X_freq = extract_band_features(X, fs)
print(f"    Frequency features: {X_freq.shape}")

# Combine features
X_combined = np.hstack([X_csp, X_freq])
print(f"    Combined: {X_combined.shape}")

# ============================================================
# 4. TRAIN CLASSIFIERS
# ============================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
lr.fit(X_train_scaled, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_scaled))
results['Logistic Regression'] = acc_lr
print(f"    LR: {acc_lr:.2%}")

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_scaled))
results['Random Forest'] = acc_rf
print(f"    RF: {acc_rf:.2%}")

# SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))
results['SVM (RBF)'] = acc_svm
print(f"    SVM: {acc_svm:.2%}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_scaled))
results['Gradient Boosting'] = acc_gb
print(f"    GB: {acc_gb:.2%}")

# Try XGBoost
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_clf.fit(X_train_scaled, y_train)
    acc_xgb = accuracy_score(y_test, xgb_clf.predict(X_test_scaled))
    results['XGBoost'] = acc_xgb
    print(f"    XGB: {acc_xgb:.2%}")
except Exception as e:
    print(f"    XGB: not available ({e})")

# ============================================================
# 5. RESULTS
# ============================================================
print("\n" + "="*60)
print("[5] RESULTS SUMMARY")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
best_name, best_acc = sorted_results[0]

print(f"\n    Best: {best_name} = {best_acc:.2%}")
print("\n    All results:")
for name, acc in sorted_results:
    print(f"        {name}: {acc:.2%}")

# Cross-validation
best_cv = None
if best_name == 'Random Forest':
    cv_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
elif best_name == 'XGBoost':
    cv_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
elif best_name == 'SVM (RBF)':
    cv_model = SVC(kernel='rbf', random_state=42)
else:
    cv_model = LogisticRegression(random_state=42, max_iter=1000)

cv_scores = cross_val_score(cv_model, X_combined, y, cv=5)
print(f"\n    5-Fold CV ({best_name}): {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

print("\n" + "="*60)
print("SUMMARY:")
print("  1. CSP features: 4 (log-variance from 2 filter pairs)")
print("  2. Added: RandomForest, SVM, XGBoost, GradientBoosting")
print("  3. Harder data: Subtle ERD, more noise, 5% label noise")
print("="*60)
