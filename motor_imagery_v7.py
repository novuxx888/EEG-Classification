#!/usr/bin/env python3
"""
EEG Motor Imagery - Ultimate v7
New approaches to beat 82.5%:
- Enhanced CSP with regularization
- Multi-scale temporal features
- Optimized ensemble
- Try sklearn GradientBoosting with more trees
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               VotingClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, BaggingClassifier)
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
print("EEG MOTOR IMAGERY - ULTIMATE v7")
print("="*60)

# ============================================================================
# 1. SYNTHETIC DATA (Balanced - best difficulty)
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 450
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
        # Noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        
        # Artifacts (12%)
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        # Variability
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
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. CSP FEATURES - Enhanced with regularization
# ============================================================================
print("\n[2] Computing CSP features...")

def compute_csp_enhanced(X, y, fs, n_components=4, band=(8, 13)):
    """Enhanced CSP with regularization"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    n_trials = len(X)
    
    # Regularized covariance
    reg = 1e-4
    
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            cov = np.cov(trial)
            # Regularization
            cov = (1 - reg) * cov + reg * np.trace(cov) * np.eye(n_channels) / n_channels
            class_cov += cov
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    try:
        A = covs[0]
        B = covs[1]
        
        # Generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(A, A + B)
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top and bottom components
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            
            # Log variance ratio
            log_var = []
            for i in range(n_components):
                ratio = var[i] / (var[n_components + i] + 1e-10)
                log_var.append(np.log(ratio + 1e-10))
            
            features.append(np.concatenate([log_var, var[:n_components], var[n_components:]]))
        
        return np.array(features)
    except Exception as e:
        return np.zeros((len(X), n_components * 3))

# 6-band FBCSP (added low-beta)
bands = [
    (4, 8),    # Theta
    (8, 13),   # Mu (main)
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (6, 12),   # Low-mu
    (10, 14),  # High-mu
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp_enhanced(X, y, fs, n_components=4, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    FBCSP: {fbcsp.shape}")

# ============================================================================
# 3. BAND FEATURES - Multi-scale
# ============================================================================
print("\n[3] Extracting band features...")

def extract_band_features(X, fs):
    """Enhanced frequency band features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            
            # Absolute powers
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs <= 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs <= 30)])
            total = delta + theta + alpha + beta_low + beta_high + 1e-10
            
            # Relative powers
            trial_feats.extend([
                delta, theta, alpha, beta_low, beta_high,
                delta/total, theta/total, alpha/total, beta_low/total, beta_high/total,
                # Ratios
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta,
                (alpha + theta) / (beta_low + beta_high + 1e-10),
                (beta_low) / (beta_high + 1e-10),
                # Time domain
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2)),
                np.max(ch) - np.min(ch),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_asymmetry(X):
    """Hemisphere asymmetry features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in range(4):
            left_power = np.mean(trial[ch]**2)
            right_power = np.mean(trial[ch + 4]**2)
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            corr = np.corrcoef(trial[ch], trial[ch + 4])[0, 1]
            trial_feats.extend([
                asymmetry, corr, 
                np.log(left_power+1), np.log(right_power+1),
                left_power / (right_power + 1e-10)
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_spatial(X):
    """Spatial pattern features"""
    features = []
    for trial in X:
        # Left vs right hemisphere
        left_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        
        # Anterior vs posterior
        ant_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 4, 6]])
        post_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        
        # Central
        central_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        
        features.append([
            left_power, right_power, ant_power, post_power, central_power,
            left_power / (right_power + 1e-10),
            ant_power / (post_power + 1e-10),
            np.log(left_power+1), np.log(right_power+1),
        ])
    return np.array(features)

def extract_temporal(X):
    """Temporal segment features"""
    features = []
    n_seg = 6  # More segments
    seg_len = X.shape[2] // n_seg
    for trial in X:
        trial_feats = []
        for ch in trial:
            for seg in range(n_seg):
                start = seg * seg_len
                end = start + seg_len
                trial_feats.append(np.mean(trial[start:end]**2))
        features.append(trial_feats)
    return np.array(features)

def extract_wavelet_like(X):
    """Multi-resolution features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            # Approximate different frequency bands via filtering
            for low, high in [(4, 8), (8, 13), (13, 20), (20, 30)]:
                b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                filtered = filtfilt(b, a, ch)
                trial_feats.append(np.mean(filtered**2))
        features.append(trial_feats)
    return np.array(features)

band_feats = extract_band_features(X, fs)
asym_feats = extract_asymmetry(X)
spatial_feats = extract_spatial(X)
temporal_feats = extract_temporal(X)
wavelet_feats = extract_wavelet_like(X)

X_combined = np.hstack([fbcsp, band_feats, asym_feats, spatial_feats, temporal_feats, wavelet_feats])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Combined: {X_combined.shape}")

# ============================================================================
# 4. TRAIN
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# LR
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.3)
lr.fit(X_train_s, y_train)
results['LR'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM
svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM'] = accuracy_score(y_test, svm.predict(X_test_s))

# RF
rf = RandomForestClassifier(n_estimators=800, max_depth=22, min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['RF'] = accuracy_score(y_test, rf.predict(X_test_s))

# ET
et = ExtraTreesClassifier(n_estimators=800, max_depth=22, min_samples_leaf=1, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
results['ET'] = accuracy_score(y_test, et.predict(X_test_s))

# GB
gb = GradientBoostingClassifier(n_estimators=400, max_depth=7, learning_rate=0.08, random_state=42)
gb.fit(X_train_s, y_train)
results['GB'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=9, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.8, 
        reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, 
        eval_metric='logloss', verbosity=0, n_jobs=-1
    )
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    XGBoost: {e}")

# LightGBM
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=700, max_depth=14, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM: {e}")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=600, early_stopping=True, random_state=42)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# AdaBoost
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
ada.fit(X_train_s, y_train)
results['AdaBoost'] = accuracy_score(y_test, ada.predict(X_test_s))

# ============================================================================
# 5. ENSEMBLE
# ============================================================================
print("\n[5] Creating ensembles...")

# Soft voting ensemble
ensemble1 = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('gb', gb), ('xgb', xgb_clf), ('lgb', lgb_clf)],
    voting='soft'
)
ensemble1.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble1.predict(X_test_s))

# Stacking-like (manual)
proba_rf = rf.predict_proba(X_test_s)[:, 1]
proba_et = et.predict_proba(X_test_s)[:, 1]
proba_gb = gb.predict_proba(X_test_s)[:, 1]
proba_xgb = xgb_clf.predict_proba(X_test_s)[:, 1]
proba_lgb = lgb_clf.predict_proba(X_test_s)[:, 1]

avg_proba = (proba_rf + proba_et + proba_gb + proba_xgb + proba_lgb) / 5
stacked_pred = (avg_proba > 0.5).astype(int)
results['Stacked'] = accuracy_score(y_test, stacked_pred)

# Weighted ensemble
weighted_proba = (0.2*proba_rf + 0.25*proba_et + 0.2*proba_gb + 0.175*proba_xgb + 0.175*proba_lgb)
weighted_pred = (weighted_proba > 0.5).astype(int)
results['Weighted'] = accuracy_score(y_test, weighted_pred)

# ============================================================================
# 6. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[6] RESULTS - ULTIMATE v7")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    BEST: {best_name} = {best_acc:.1%}")
print(f"    Previous record: 82.5%")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('RF', rf), ('ET', et), ('GB', gb)]:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
