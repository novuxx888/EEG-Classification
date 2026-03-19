#!/usr/bin/env python3
"""
EEG Motor Imagery - Harder Synthetic Data + Best Methods
- 3 classes (left/right/rest) - much harder
- 16 channels - more spatial info
- 5% suppression - very subtle
- Best classifiers
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
print("EEG MOTOR IMAGERY - HARDER DATA (3-class)")
print("="*60)

# ============================================================================
# 1. CREATE HARDER SYNTHETIC DATA - 3 CLASS
# ============================================================================
print("\n[1] Creating harder 3-class synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 600  # More trials for 3 classes
n_channels = 16  # More channels

X = []
y = []

for trial in range(n_trials):
    # 3 classes: 0=left, 1=right, 2=rest
    label = np.random.randint(0, 3)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
        # More realistic noise
        white_noise = np.random.randn(len(t)) * 12  # More noise
        drift = np.linspace(0, 3, len(t)) * np.random.randn() * 4  # More drift
        # Random artifacts (15%)
        if np.random.rand() < 0.15:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 30
        
        base += white_noise + drift
        
        # More trial/channel variability
        trial_factor = np.random.uniform(0.3, 1.7)
        ch_factor = np.random.uniform(0.6, 1.4)
        base *= trial_factor * ch_factor
        
        # 50% show effect, 5% suppression (very subtle!)
        show_effect = np.random.rand() < 0.50
        suppression = 0.95 if show_effect else 1.0
        
        # Channel layout:
        # Left motor cortex (C3): channels 4-7
        # Right motor cortex (C4): channels 8-11
        # Others: channels 0-3, 12-15
        
        if label == 0:  # LEFT - suppress right motor cortex (C4)
            if ch in [8, 9, 10, 11]:  # Right hemisphere
                base *= (0.95 + np.random.uniform(-0.08, 0.08)) * suppression
        elif label == 1:  # RIGHT - suppress left motor cortex (C3)
            if ch in [4, 5, 6, 7]:  # Left hemisphere
                base *= (0.95 + np.random.uniform(-0.08, 0.08)) * suppression
        # Rest (label=2): no suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}, Rest={np.sum(y==2)}")
print(f"    Difficulty: 3-class, 50% effect, 5% suppression, {n_channels} channels")

# ============================================================================
# 2. COMPUTE CSP + FBCSP FEATURES
# ============================================================================
print("\n[2] Computing CSP + FBCSP features...")

def compute_csp_multiclass(X, y, fs, n_components=3, band=(8, 13)):
    """Enhanced CSP for multi-class using one-vs-rest"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    n_classes = len(np.unique(y))
    
    # One-vs-rest CSP for each class
    all_features = []
    
    for class_idx in range(n_classes):
        # Binary classification: this class vs rest
        y_binary = (y == class_idx).astype(int)
        
        covs = []
        for c in [0, 1]:
            class_cov = np.zeros((n_channels, n_channels))
            for trial in X_filt[y_binary == c]:
                cov = np.cov(trial)
                class_cov += cov / (np.trace(cov) + 1e-10)
            class_cov /= np.sum(y_binary == c) + 1e-10
            covs.append(class_cov)
        
        try:
            reg = 1e-5
            A = covs[0] + reg * np.eye(n_channels)
            B = covs[1] + reg * np.eye(n_channels)
            C = A + B + 1e-10*np.eye(n_channels)
            
            eigenvalues, eigenvectors = eigh(np.linalg.inv(C) @ A)
            idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
            eigenvectors = eigenvectors[:, idx]
            W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
            
            class_features = []
            for trial in X_filt:
                projected = W.T @ trial
                var = np.var(projected, axis=1)
                trial_feat = np.concatenate([
                    np.log(var[:n_components] / (var[n_components:] + 1e-10)),
                    var
                ])
                class_features.append(trial_feat)
            
            all_features.append(np.array(class_features))
        except:
            all_features.append(np.zeros((len(X), n_components * 2)))
    
    # Concatenate all class features
    csp_features = np.hstack(all_features)
    return csp_features

# 5-band FBCSP
bands = [(4, 8), (8, 13), (13, 20), (20, 30), (6, 12)]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp_multiclass(X, y, fs, n_components=3, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    FBCSP features: {fbcsp.shape}")

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Extracting additional features...")

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

def extract_motor_asymmetry(X):
    """Motor cortex specific asymmetry"""
    features = []
    # C3: channels 4-7, C4: channels 8-11
    for trial in X:
        trial_feats = []
        
        c3_power = np.mean([np.mean(trial[ch]**2) for ch in [4, 5, 6, 7]])
        c4_power = np.mean([np.mean(trial[ch]**2) for ch in [8, 9, 10, 11]])
        
        # Asymmetry
        asymmetry = (c3_power - c4_power) / (c3_power + c4_power + 1e-10)
        
        # Other channels
        other_power = np.mean([np.mean(trial[ch]**2) for ch in list(range(4)) + list(range(12, 16))])
        
        trial_feats.extend([
            c3_power, c4_power, asymmetry, c3_power / (c4_power + 1e-10),
            other_power, c3_power / (other_power + 1e-10), c4_power / (other_power + 1e-10)
        ])
        features.append(trial_feats)
    return np.array(features)

def extract_temporal_features(X):
    features = []
    n_seg = 5
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

band_features = extract_band_features(X, fs)
motor_asym = extract_motor_asymmetry(X)
temporal_features = extract_temporal_features(X)

X_combined = np.hstack([fbcsp, band_features, motor_asym, temporal_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Combined features: {X_combined.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS
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
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.5, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM
svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest
rf = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees
et = ExtraTreesClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.08, 
                                 random_state=42, use_label_encoder=False, 
                                 eval_metric='mlogloss', verbosity=0, n_jobs=-1)
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except:
    pass

# LightGBM
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(n_estimators=600, max_depth=12, learning_rate=0.06, 
                                 objective='multiclass', random_state=42, verbose=-1, n_jobs=-1)
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except:
    pass

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=600, 
                    early_stopping=True, random_state=42)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[5] RESULTS - HARD 3-CLASS DATA")
print("="*60)
print(f"    (Chance = 33.3%)")

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    above_chance = f" (+{acc-0.333:.1%})" if acc > 0.333 else ""
    print(f"    {name}: {acc:.1%}{above_chance}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} = {best_acc:.1%}")
print(f"    Previous hard data best: ~52% (also 3-class)")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clfs = [('RF', rf), ('ET', et), ('XGBoost', xgb_clf)]

for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
