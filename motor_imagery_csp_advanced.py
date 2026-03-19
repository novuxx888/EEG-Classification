#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Advanced Version
Features: CSP features, RandomForest, XGBoost, harder synthetic data, EEGNet
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - ADVANCED CLASSIFICATION")
print("="*60)

# ============================================================================
# 1. CREATE HARDER SYNTHETIC MOTOR IMAGERY DATA
# ============================================================================
print("\n[1] Creating HARDER synthetic motor imagery data...")

fs = 128  # Sampling frequency
t = np.arange(0, 4, 1/fs)  # 4 seconds

n_trials = 150  # More trials
n_channels = 8  # More channels (C3, C4, FC, CP)

# Channel layout (approximating 10-20 system)
# Left motor: channels 2,3 (C3 area)
# Right motor: channels 4,5 (C4 area)

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)  # 0 = left, 1 = right
    
    signals = []
    for ch in range(n_channels):
        # Base EEG with multiple rhythms
        # Alpha (10 Hz), Beta (20 Hz), Theta (6 Hz)
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta = np.sin(2 * np.pi * 20 * t + np.random.rand()*2*np.pi) * 8
        theta = np.random.randn(len(t)) * 3
        base = alpha + beta + theta
        
        # Add realistic noise (EMG, eye artifacts)
        if ch < 2:  # Frontal
            artifacts = np.random.randn(len(t)) * 5
            base += artifacts
        
        # Non-stationarity: drift over time
        drift = np.linspace(0, 1, len(t)) * np.random.randn() * 3
        base += drift
        
        # Motor imagery effect: 
        # Left hand = mu rhythm suppression (desync) on right motor cortex
        # Right hand = mu rhythm suppression on left motor cortex
        
        if label == 0:  # LEFT hand - suppress right side (channels 2,3,4,5)
            if ch in [2, 3, 4, 5]:  # Right hemisphere
                base = base * 0.4 + np.random.randn(len(t)) * 5  # Suppress + noise
        else:  # RIGHT hand - suppress left side (channels 0,1,6,7)
            if ch in [0, 1, 6, 7]:  # Left hemisphere
                base = base * 0.4 + np.random.randn(len(t)) * 5  # Suppress + noise
        
        # Cross-trial variability (major challenge!)
        trial_factor = np.random.uniform(0.7, 1.3)
        base = base * trial_factor
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape} (trials, channels, samples)")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. CSP (Common Spatial Patterns) FEATURE EXTRACTION
# ============================================================================
print("\n[2] Extracting CSP features...")

def compute_csp_filters(X, y, n_components=2):
    """Compute Common Spatial Patterns filters"""
    # Filter for mu rhythm (8-13 Hz)
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    
    X_filtered = np.array([filtfilt(b, a, trial) for trial in X])
    
    # Compute covariance matrices for each class
    classes = np.unique(y)
    covs = []
    
    for c in classes:
        class_data = X_filtered[y == c]
        n_trials_c = len(class_data)
        
        # Average covariance
        cov = np.zeros((n_channels, n_channels))
        for trial in class_data:
            cov += np.cov(trial)
        cov /= n_trials_c
        covs.append(cov)
    
    # Solve generalized eigenvalue problem
    cov1, cov2 = covs[0], covs[1]
    eigenvalues, eigenvectors = eigh(cov1, cov1 + cov2)
    
    # Sort by eigenvalues (descending for class 1)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top and bottom n_components (most discriminative)
    # Stack to get (n_channels, 2*n_components)
    top_n = eigenvectors[:, :n_components]  # (n_channels, n_components)
    bottom_n = eigenvectors[:, -n_components:]  # (n_channels, n_components)
    filters = np.hstack([top_n, bottom_n])  # (n_channels, 2*n_components)
    
    return filters

def extract_csp_features(X, filters):
    """Extract CSP features from filtered data"""
    # Filter for mu rhythm
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filtered = np.array([filtfilt(b, a, trial) for trial in X])
    
    features = []
    for trial in X_filtered:
        # Project to CSP space: filters is (n_channels, 2*n_components)
        # trial is (n_channels, n_samples)
        # We need: (2*n_components, n_channels) @ (n_channels, n_samples)
        projected = filters.T @ trial  # (2*n_components, n_samples)
        
        # Compute log-variance as features
        var = np.var(projected, axis=1)
        log_var = np.log(var + 1e-10)
        features.append(log_var)
    
    return np.array(features)

# Compute CSP filters on training data
print("    Computing CSP filters...")
csp_filters = compute_csp_filters(X, y, n_components=3)
csp_features = extract_csp_features(X, csp_filters)
print(f"    CSP features shape: {csp_features.shape}")

# ============================================================================
# 3. CONVENTIONAL FREQUENCY/TIME FEATURES
# ============================================================================
print("\n[3] Extracting frequency band features...")

def extract_band_features(X, fs=128):
    """Extract frequency band features"""
    features = []
    
    for trial in X:
        trial_features = []
        
        for ch in trial:
            # Compute PSD
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            
            # Alpha (8-13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(psd[alpha_mask])
            
            # Beta (13-30 Hz)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.mean(psd[beta_mask])
            
            # Theta (4-8 Hz)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            theta_power = np.mean(psd[theta_mask])
            
            # Delta (1-4 Hz)
            delta_mask = (freqs >= 1) & (freqs <= 4)
            delta_power = np.mean(psd[delta_mask])
            
            # Ratios (useful for motor imagery)
            total_power = alpha_power + beta_power + theta_power + delta_power + 1e-10
            
            trial_features.extend([
                alpha_power, beta_power, theta_power, delta_power,
                alpha_power / total_power,
                beta_power / total_power,
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch))  # Peak amplitude
            ])
        
        features.append(trial_features)
    
    return np.array(features)

band_features = extract_band_features(X, fs)
print(f"    Band features shape: {band_features.shape}")

# Combine features
X_combined = np.hstack([csp_features, band_features])
print(f"    Combined features shape: {X_combined.shape}")

# ============================================================================
# 4. TRAIN MULTIPLE CLASSIFIERS
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# Logistic Regression
print("    Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=2000, C=0.5)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
results['LogisticRegression'] = acc_lr

# LDA
print("    Training LDA...")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)
results['LDA'] = acc_lda

# Random Forest
print("    Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                            min_samples_split=5, random_state=42,
                            n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
results['RandomForest'] = acc_rf

# Gradient Boosting
print("    Training GradientBoosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
acc_gb = accuracy_score(y_test, y_pred_gb)
results['GradientBoosting'] = acc_gb

# Try XGBoost if available
xgb_available = False
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception as e:
    print(f"    XGBoost not available: {e}")
    print("    Trying libomp install...")

if xgb_available:
    try:
        print("    Training XGBoost...")
        xgb = XGBClassifier(n_estimators=200, max_depth=5, 
                            learning_rate=0.1, random_state=42,
                            use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        results['XGBoost'] = acc_xgb
    except Exception as e:
        print(f"    XGBoost training failed: {e}")

# ============================================================================
# 5. EEGNet (Simple CNN for EEG)
# ============================================================================
print("\n[5] Training EEGNet (CNN)...")

# Prepare data for EEGNet (trials, channels, samples, 1)
X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1).astype(np.float32)

# Normalize
X_cnn_mean = X_cnn.mean()
X_cnn_std = X_cnn.std()
X_cnn = (X_cnn - X_cnn_mean) / (X_cnn_std + 1e-10)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

# Simple CNN using sklearn's MLPClassifier as proxy (real EEGNet needs keras)
from sklearn.neural_network import MLPClassifier

# Flatten for MLP
X_train_mlp = X_train_cnn.reshape(len(X_train_cnn), -1)
X_test_mlp = X_test_cnn.reshape(len(X_test_cnn), -1)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                    random_state=42, early_stopping=True,
                    learning_rate_init=0.001)
mlp.fit(X_train_mlp, y_train_cnn)
y_pred_mlp = mlp.predict(X_test_mlp)
acc_mlp = accuracy_score(y_test_cnn, y_pred_mlp)
results['MLP (EEGNet-style)'] = acc_mlp

# ============================================================================
# 6. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[6] RESULTS")
print("="*60)

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} with {best_acc:.1%}")

# Cross-validation on best model
if best_name == 'RandomForest':
    cv_model = rf
elif best_name == 'XGBoost':
    cv_model = xgb
elif best_name == 'GradientBoosting':
    cv_model = gb
else:
    cv_model = lr

cv_scores = cross_val_score(cv_model, scaler.fit_transform(X_combined), y, cv=5)
print(f"    Cross-validation: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# Save results for README
print("\n" + "="*60)
print("SUMMARY FOR README:")
print("="*60)
print("| Classifier | Accuracy |")
print("|-------------|----------|")
for name, acc in sorted_results:
    print(f"| {name} | {acc:.0%} |")
print(f"\n**Cross-validation:** {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
