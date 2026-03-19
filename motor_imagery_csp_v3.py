#!/usr/bin/env python3
"""
EEG Motor Imagery - CSP Enhanced + Tuned RF/XGBoost
- Improved CSP with multiple regularization options
- Better tuned RandomForest and XGBoost
- Feature selection
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - CSP ENHANCED + RF/XGBoost")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA - SLIGHTLY HARDER
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 450  # More trials
n_channels = 10  # More channels for spatial resolution

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG (more realistic)
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 3
        base = alpha + beta1 + beta2 + theta + delta
        
        # More realistic noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        # Random artifacts (12%)
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # 55% show effect (slightly harder), 13% suppression
        show_effect = np.random.rand() < 0.55
        suppression = 0.87 if show_effect else 1.0
        
        # Left motor cortex: channels 3,4,5 (C3)
        # Right motor cortex: channels 5,6,7 (C4)
        # More realistic channel mapping
        if label == 0:  # LEFT - suppress right motor cortex (C4)
            if ch in [5, 6, 7, 8]:  # Right hemisphere
                base *= (0.87 + np.random.uniform(-0.1, 0.1)) * suppression
        else:  # RIGHT - suppress left motor cortex (C3)
            if ch in [2, 3, 4, 9]:  # Left hemisphere
                base *= (0.87 + np.random.uniform(-0.1, 0.1)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: 55% effect, 13% suppression, {n_channels} channels")

# ============================================================================
# 2. ENHANCED CSP + FBCSP FEATURES
# ============================================================================
print("\n[2] Computing enhanced CSP + FBCSP features...")

def compute_csp_enhanced(X, y, fs, n_components=4, band=(8, 13)):
    """Enhanced CSP with regularization"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    # Compute spatial covariance matrices
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            # Regularized covariance
            cov = np.cov(trial)
            reg = 1e-5 * np.trace(cov) * np.eye(n_channels)
            class_cov += (cov + reg) / (np.sum(y == c) + 1e-10)
        covs.append(class_cov)
    
    try:
        # CSP with regularization
        reg_param = 1e-5
        A = covs[0] + reg_param * np.eye(n_channels)
        B = covs[1] + reg_param * np.eye(n_channels)
        C = A + B + 1e-10*np.eye(n_channels)
        
        eigenvalues, eigenvectors = eigh(np.linalg.inv(C) @ A)
        
        # Sort eigenvalues (closest to 0.5 = best for discrimination)
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top and bottom components
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        # Project and compute features
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            
            # Log variance features
            log_var = np.log(var[:n_components] / (var[n_components:] + 1e-10))
            log_var = np.nan_to_num(log_var, nan=0.0, posinf=0.0, neginf=0.0)
            
            trial_feat = np.concatenate([
                log_var,  # CSP features
                var,      # Variance features
                var[:n_components] / (var[n_components:] + 1e-10),  # Ratio features
            ])
            features.append(trial_feat)
        
        return np.array(features)
    except:
        return np.zeros((len(X), n_components * 4))

# 6-band FBCSP (expanded)
bands = [
    (4, 8),    # Theta
    (8, 13),   # Mu (main)
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (6, 12),   # Low mu
    (10, 14),  # High alpha
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp_enhanced(X, y, fs, n_components=4, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    FBCSP features: {fbcsp.shape}")

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Extracting additional features...")

def extract_band_features(X, fs):
    """Enhanced frequency band power features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=64)
            
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs <= 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs <= 30)])
            total = delta + theta + alpha + beta_low + beta_high + 1e-10
            
            # Absolute and relative powers
            trial_feats.extend([
                delta, theta, alpha, beta_low, beta_high,
                delta/total, theta/total, alpha/total, beta_low/total, beta_high/total,
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta,
                (alpha + theta) / (beta_low + beta_high + 1e-10),
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2)),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_asymmetry(X):
    """Hemisphere asymmetry features - enhanced"""
    features = []
    # Left channels: 0-4, Right channels: 5-9
    for trial in X:
        trial_feats = []
        for i in range(5):
            left_power = np.mean(trial[i]**2)
            right_power = np.mean(trial[i+5]**2)
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            corr = np.corrcoef(trial[i], trial[i+5])[0, 1]
            trial_feats.extend([
                asymmetry, corr, 
                np.log(left_power+1), np.log(right_power+1),
                left_power / (right_power + 1e-10)
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_spatial_features(X):
    """Spatial patterns features"""
    features = []
    # Left: 0-4, Right: 5-9
    for trial in X:
        trial_feats = []
        left_power = np.mean([np.mean(trial[ch]**2) for ch in range(5)])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in range(5, 10)])
        ant_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 5, 7]])
        post_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 4, 6, 8, 9]])
        central_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 4, 5, 6, 7]])
        
        # Motor cortex specific
        c3_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 4, 9]])  # C3
        c4_power = np.mean([np.mean(trial[ch]**2) for ch in [5, 6, 7, 8]])  # C4
        
        trial_feats.extend([
            left_power, right_power, ant_power, post_power, central_power,
            left_power / (right_power + 1e-10),
            ant_power / (post_power + 1e-10),
            c3_power, c4_power, c3_power / (c4_power + 1e-10)
        ])
        features.append(trial_feats)
    return np.array(features)

def extract_temporal_features(X):
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

def extract_connectivity_features(X):
    """Connectivity features between channels"""
    features = []
    n_pairs = min(10, X.shape[1] // 2)
    for trial in X:
        trial_feats = []
        for i in range(n_pairs):
            corr = np.corrcoef(trial[i], trial[i+X.shape[1]//2])[0, 1]
            trial_feats.append(corr if not np.isnan(corr) else 0)
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
asym_features = extract_asymmetry(X)
spatial_features = extract_spatial_features(X)
temporal_features = extract_temporal_features(X)
connectivity_features = extract_connectivity_features(X)

X_combined = np.hstack([
    fbcsp, band_features, asym_features, 
    spatial_features, temporal_features, connectivity_features
])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Combined features: {X_combined.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS WITH TUNING
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.5, solver='lbfgs')
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM with tuning
svm = SVC(kernel='rbf', C=2.5, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest - tuned
rf = RandomForestClassifier(
    n_estimators=800, 
    max_depth=22, 
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees - tuned
et = ExtraTreesClassifier(
    n_estimators=800, 
    max_depth=22,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42, 
    n_jobs=-1
)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# Gradient Boosting - tuned
gb = GradientBoostingClassifier(
    n_estimators=350, 
    max_depth=7, 
    learning_rate=0.08,
    subsample=0.85,
    random_state=42
)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost - tuned
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    )
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    XGBoost error: {e}")

# LightGBM - tuned
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=700, 
        max_depth=14, 
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42, 
        verbose=-1,
        n_jobs=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM error: {e}")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=600, 
                    early_stopping=True, random_state=42)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# ============================================================================
# 5. ENSEMBLE
# ============================================================================
print("\n[5] Creating ensemble...")

ensemble_estimators = [
    ('svm', svm), 
    ('rf', rf), 
    ('et', et),
    ('gb', gb),
    ('mlp', mlp)
]
if 'xgb_clf' in dir():
    ensemble_estimators.append(('xgb', xgb_clf))
if 'lgb_clf' in dir():
    ensemble_estimators.append(('lgb', lgb_clf))

ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 6. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[6] RESULTS")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} = {best_acc:.1%}")
print(f"    Previous best: 82.5%")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clfs = [
    ('RF', rf), 
    ('ET', et), 
    ('GB', gb)
]
if 'xgb_clf' in dir():
    cv_clfs.append(('XGBoost', xgb_clf))
if 'lgb_clf' in dir():
    cv_clfs.append(('LightGBM', lgb_clf))

for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
