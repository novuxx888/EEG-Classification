#!/usr/bin/env python3
"""
EEG Motor Imagery - Wavelet-based Features
Using Continuous Wavelet Transform for better time-frequency representation
"""

import numpy as np
from scipy.signal import butter, filtfilt, cwt, morlet2
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - WAVELET FEATURES")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA (medium difficulty)
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 400
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 16
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 3
        
        base = alpha + beta1 + beta2 + theta + delta
        
        # Noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        if np.random.rand() < 0.10:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 20
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # 55% show effect, 12% suppression (medium difficulty)
        show_effect = np.random.rand() < 0.55
        suppression = 0.88 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.88 + np.random.uniform(-0.06, 0.06)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.88 + np.random.uniform(-0.06, 0.06)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. WAVELET-BASED FEATURE EXTRACTION
# ============================================================================
print("\n[2] Computing Wavelet features...")

def extract_wavelet_features(X, fs):
    """Extract features using Continuous Wavelet Transform with Morlet wavelet"""
    features = []
    
    # Wavelet frequencies (covering relevant EEG bands)
    freqs = np.array([2, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    widths = fs * 10 / (2 * np.pi * freqs)  # Convert to wavelet widths
    
    for trial in X:
        trial_features = []
        
        for ch in trial:
            # Compute CWT
            cwt_matrix = cwt(ch, morlet2, widths)
            
            # Band-wise features (average power in each band)
            for i, freq in enumerate(freqs):
                band_power = np.mean(np.abs(cwt_matrix[i, :])**2)
                trial_features.append(band_power)
            
            # Time-frequency features (divide into 4 time segments)
            n_segments = 4
            segment_size = cwt_matrix.shape[1] // n_segments
            for seg in range(n_segments):
                seg_power = np.mean(np.abs(cwt_matrix[:, seg*segment_size:(seg+1)*segment_size])**2)
                trial_features.append(seg_power)
            
            # Alpha band time evolution (important for motor imagery)
            alpha_idx = list(freqs).index(10)
            alpha_power = np.abs(cwt_matrix[alpha_idx, :])**2
            trial_features.append(np.mean(alpha_power[:len(t)//2]))
            trial_features.append(np.mean(alpha_power[len(t)//2:]))
            trial_features.append(np.std(alpha_power))
        
        features.append(trial_features)
    
    return np.array(features)

def extract_wavelet_csp_features(X, y, fs, n_components=3):
    """Wavelet + CSP hybrid features"""
    # Filter bank
    bands = [(8, 13), (13, 20), (20, 30)]
    csp_features = []
    
    for trial in X:
        trial_csp = []
        
        for b_start, b_end in bands:
            # Bandpass filter
            b, a = butter(4, [b_start/(fs/2), b_end/(fs/2)], btype='band')
            filtered = np.array([filtfilt(b, a, ch) for ch in trial])
            
            # CSP for this band
            # Compute covariance
            cov_left = []
            cov_right = []
            
            for i, (trial_data, label) in enumerate(zip(X, y)):
                trial_filtered = []
                for ch_idx in range(trial_data.shape[0]):
                    b_f, a_f = butter(4, [b_start/(fs/2), b_end/(fs/2)], btype='band')
                    trial_filtered.append(filtfilt(b_f, a_f, trial_data[ch_idx]))
                trial_filtered = np.array(trial_filtered)
                
                cov = np.cov(trial_filtered)
                if label == 0:
                    cov_left.append(cov)
                else:
                    cov_right.append(cov)
            
            avg_cov_left = np.mean(cov_left, axis=0)
            avg_cov_right = np.mean(cov_right, axis=0)
            
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = eigh(avg_cov_left, avg_cov_left + avg_cov_right)
            
            # Get top and bottom eigenvectors (most discriminative)
            idx = np.argsort(eigenvalues)
            for i in range(n_components):
                v_top = eigenvectors[:, idx[-1-i]]
                v_bottom = eigenvectors[:, idx[i]]
                
                # Project current trial
                projected = filtered.T @ v_top
                trial_csp.append(np.var(projected))
                trial_csp.append(np.var(filtered.T @ v_bottom))
        
        csp_features.append(trial_csp)
    
    return np.array(csp_features)

# Extract wavelet features
print("    Computing wavelet transform features...")
X_wavelet = extract_wavelet_features(X, fs)
print(f"    Wavelet features shape: {X_wavelet.shape}")

# Also extract standard band features
print("    Computing standard band features...")
def extract_band_features(X, fs):
    """Extract frequency band power features"""
    features = []
    bands = {'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 13), 'beta1': (13, 20), 'beta2': (20, 30)}
    
    for trial in X:
        trial_features = []
        
        for ch in trial:
            for band_name, (low, high) in bands.items():
                b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                filtered = filtfilt(b, a, ch)
                trial_features.append(np.mean(filtered**2))
        
        features.append(trial_features)
    
    return np.array(features)

X_band = extract_band_features(X, fs)

# Combine features
X_combined = np.hstack([X_wavelet, X_band])
print(f"    Combined features shape: {X_combined.shape}")

# ============================================================================
# 3. TRAIN AND EVALUATE
# ============================================================================
print("\n[3] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

# RandomForest
rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results['RandomForest'] = accuracy_score(y_test, y_pred)

# GradientBoosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
results['GradientBoosting'] = accuracy_score(y_test, y_pred)

# ExtraTrees
et = ExtraTreesClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
results['ExtraTrees'] = accuracy_score(y_test, y_pred)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
results['MLP'] = accuracy_score(y_test, y_pred)

# SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
results['SVM-RBF'] = accuracy_score(y_test, y_pred)

print("\n[4] Results:")
print("-" * 40)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"    {name}: {acc:.2%}")

best_name = max(results, key=results.get)
best_acc = results[best_name]

# Cross-validation
print("\n[5] Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42), 
                        X_combined, y, cv=cv, scoring='accuracy')
print(f"    RandomForest CV: {rf_cv.mean():.2%} ± {rf_cv.std():.2%}")

print("\n" + "="*60)
print(f"BEST: {best_name} at {best_acc:.2%}")
print("="*60)

# Save results
with open('/Users/lobter/.openclaw/workspace/EEG-Classification/results_wavelet.txt', 'w') as f:
    f.write("Wavelet Features Results\n")
    f.write("="*50 + "\n")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"{name}: {acc:.2%}\n")
    f.write(f"\nBest: {best_name} at {best_acc:.2%}\n")
    f.write(f"CV: {rf_cv.mean():.2%} ± {rf_cv.std():.2%}\n")

print("\nResults saved to results_wavelet.txt")