#!/usr/bin/env python3
"""
EEG Motor Imagery - STREAMLINED v6
Focus on what works: CSP features + RF/ET + optimized ensemble
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - STREAMLINED v6")
print("="*60)

# ============================================================================
# 1. SYNTHETIC DATA
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
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        show_effect = np.random.rand() < 0.60
        suppression = 0.86 if show_effect else 1.0
        
        if label == 0:  # LEFT
            if ch in [2, 3, 6, 7]:
                base *= (0.86 + np.random.uniform(-0.1, 0.1)) * suppression
        else:  # RIGHT
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
# 2. CSP FEATURES
# ============================================================================
print("\n[2] Computing CSP features...")

def compute_csp(X, y, fs, n_components=3, band=(8, 13)):
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            class_cov += np.cov(trial)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    try:
        reg = 1e-5
        eigenvalues, eigenvectors = eigh(covs[0] + reg*np.eye(n_channels), 
                                          covs[0] + covs[1] + reg*np.eye(n_channels))
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, idx]
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            log_var = np.log(var[:n_components] / (var[n_components:] + 1e-10) + 1e-10)
            features.append(np.concatenate([log_var, var]))
        return np.array(features)
    except:
        return np.zeros((len(X), n_components * 4))

# 5-band FBCSP
bands = [(4, 8), (8, 13), (13, 20), (20, 30), (6, 12)]
fbcsp = []
for band in bands:
    csp = compute_csp(X, y, fs, n_components=3, band=band)
    fbcsp.append(csp)
fbcsp = np.hstack(fbcsp)
fbcsp = np.nan_to_num(fbcsp)
print(f"    FBCSP: {fbcsp.shape}")

# ============================================================================
# 3. BAND FEATURES
# ============================================================================
print("\n[3] Extracting band features...")

def extract_band_features(X, fs):
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
            
            trial_feats.extend([delta, theta, alpha, beta_low, beta_high])
            trial_feats.extend([delta/total, theta/total, alpha/total, beta_low/total, beta_high/total])
            trial_feats.extend([
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta, (alpha + theta)/(beta_low + beta_high + 1e-10)
            ])
            trial_feats.extend([
                np.mean(ch), np.std(ch), np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2))
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_hemi_features(X):
    features = []
    for trial in X:
        left = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        asym = (left - right) / (left + right + 1e-10)
        features.append([
            np.log(left+1), np.log(right+1), left/(right+1e-10), asym,
            np.mean(trial[0]**2), np.mean(trial[1]**2),
            np.mean(trial[2]**2), np.mean(trial[3]**2)
        ])
    return np.array(features)

def extract_temporal(X):
    features = []
    n_seg = 5
    seg_len = X.shape[2] // n_seg
    for trial in X:
        trial_feats = []
        for ch in trial:
            for seg in range(n_seg):
                trial_feats.append(np.mean(trial[seg*seg_len:(seg+1)*seg_len]**2))
        features.append(trial_feats)
    return np.array(features)

band_feats = extract_band_features(X, fs)
hemi_feats = extract_hemi_features(X)
temporal_feats = extract_temporal(X)

X_combined = np.hstack([fbcsp, band_feats, hemi_feats, temporal_feats])
X_combined = np.nan_to_num(X_combined)
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
lr = LogisticRegression(random_state=42, max_iter=2000, C=0.5)
lr.fit(X_train_s, y_train)
results['LR'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM
svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM'] = accuracy_score(y_test, svm.predict(X_test_s))

# RF
rf = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['RF'] = accuracy_score(y_test, rf.predict(X_test_s))

# ET
et = ExtraTreesClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
results['ET'] = accuracy_score(y_test, et.predict(X_test_s))

# GB
gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
results['GB'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, random_state=42, 
        use_label_encoder=False, eval_metric='logloss', verbosity=0, n_jobs=-1)
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    XGBoost: {e}")

# LightGBM
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(n_estimators=600, max_depth=12, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=-1)
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM: {e}")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True, random_state=42)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# Ensemble
ensemble = VotingClassifier(estimators=[('svm', svm), ('rf', rf), ('et', et), ('gb', gb)], voting='soft')
ensemble.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[5] RESULTS - STREAMLINED v6")
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

for name, clf in [('RF', rf), ('ET', et), ('GB', gb), ('XGB', xgb_clf if 'xgb_clf' in dir() else None)]:
    if clf is not None:
        cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
        print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
