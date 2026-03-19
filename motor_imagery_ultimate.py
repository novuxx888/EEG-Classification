#!/usr/bin/env python3
"""
EEG Motor Imagery - Ultimate Version
Aims to beat 82.5% record with:
- Enhanced 7-band FBCSP
- More CSP components
- Better tuned classifiers (CatBoost, optimized LightGBM)
- Enhanced spatial features
- Deeper EEGNet
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - ULTIMATE VERSION")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA (Slightly harder)
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 500  # More trials
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG with more realism
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 3
        base = alpha + beta1 + beta2 + theta + delta
        
        # Noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        # Artifacts
        if np.random.rand() < 0.10:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 20
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # 55% show effect, 16% suppression (slightly harder than before)
        show_effect = np.random.rand() < 0.55
        suppression = 0.84 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.84 + np.random.uniform(-0.12, 0.12)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.84 + np.random.uniform(-0.12, 0.12)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: 55% effect, 16% suppression")

# ============================================================================
# 2. ENHANCED CSP + 7-BAND FBCSP
# ============================================================================
print("\n[2] Computing Enhanced CSP + 7-band FBCSP...")

def compute_csp_for_band(X, y, fs, n_components=4, band=(8, 13)):
    """Enhanced CSP with more components"""
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
                var,
                var[:n_components] / (var[:n_components].sum() + 1e-10),
                var[n_components:] / (var[n_components:].sum() + 1e-10),
            ])
            features.append(trial_feat)
        
        return np.array(features)
    except:
        return np.zeros((len(X), n_components * 4))

# 7-band FBCSP (added delta and gamma)
bands = [
    (2, 4),    # Delta
    (4, 8),    # Theta
    (8, 13),   # Mu
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (6, 12),   # Low mu
    (30, 45),  # Gamma
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp_for_band(X, y, fs, n_components=4, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
print(f"    FBCSP total: {fbcsp.shape}")

# Handle NaN/Inf
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# 3. ENHANCED FEATURES
# ============================================================================
print("\n[3] Extracting enhanced features...")

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
            gamma = np.mean(psd[(freqs >= 30) & (freqs <= 45)])
            total = delta + theta + alpha + beta_low + beta_high + gamma + 1e-10
            
            # Power ratios
            trial_feats.extend([
                delta, theta, alpha, beta_low, beta_high, gamma,
                # Relative powers
                delta/total, theta/total, alpha/total, beta_low/total, beta_high/total, gamma/total,
                # Key ratios
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta,
                (alpha + theta) / (beta_low + beta_high + 1e-10),
                theta/alpha,
                (beta_low + beta_high) / (alpha + 1e-10),
                # Time domain
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2)),
                # Additional
                np.max(ch) - np.min(ch),
                np.sum(ch > np.mean(ch) + np.std(ch)),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_asymmetry(X):
    """Enhanced hemisphere asymmetry features"""
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
                left_power / (right_power + 1e-10),
                np.abs(asymmetry),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_spatial_features(X):
    """Enhanced spatial patterns features"""
    features = []
    for trial in X:
        trial_feats = []
        # Hemisphere powers
        left_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        ant_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 4, 6]])
        post_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        
        # Motor cortex specific (C3=ch1, C4=ch5 roughly)
        c3 = np.mean(trial[1]**2)  # Left motor
        c4 = np.mean(trial[5]**2)  # Right motor
        
        trial_feats.extend([
            left_power, right_power, ant_power, post_power,
            left_power / (right_power + 1e-10),
            ant_power / (post_power + 1e-10),
            c3, c4,
            c3 / (c4 + 1e-10),
            np.log(c3 + 1), np.log(c4 + 1),
        ])
        features.append(trial_feats)
    return np.array(features)

def extract_temporal_features(X):
    """Enhanced temporal segment features"""
    features = []
    n_seg = 8  # More segments
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
    for trial in X:
        trial_feats = []
        # Correlation matrix as features
        corr = np.corrcoef(trial)
        # Upper triangle of correlation matrix
        idx = np.triu_indices(n_channels, k=1)
        trial_feats.extend(corr[idx])
        # Coherence-like features
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                trial_feats.append(np.abs(corr[i,j]))
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
asym_features = extract_asymmetry(X)
spatial_features = extract_spatial_features(X)
temporal_features = extract_temporal_features(X)
connectivity_features = extract_connectivity_features(X)

X_combined = np.hstack([fbcsp, band_features, asym_features, spatial_features, temporal_features, connectivity_features])
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
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.5, solver='lbfgs')
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM with tuning
svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest
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

# ExtraTrees
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

# Gradient Boosting with more trees
gb = GradientBoostingClassifier(
    n_estimators=400, 
    max_depth=7, 
    learning_rate=0.08,
    subsample=0.85,
    random_state=42
)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
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

# LightGBM
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

# CatBoost
print("    Training CatBoost...")
try:
    from catboost import CatBoostClassifier
    cat_clf = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_clf.fit(X_train_s, y_train)
    results['CatBoost'] = accuracy_score(y_test, cat_clf.predict(X_test_s))
except Exception as e:
    print(f"    CatBoost error: {e}")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=700, 
                    early_stopping=True, random_state=42, alpha=0.001)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# AdaBoost
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
ada.fit(X_train_s, y_train)
results['AdaBoost'] = accuracy_score(y_test, ada.predict(X_test_s))

# ============================================================================
# 5. ENHANCED EEGNet
# ============================================================================
print("\n[5] Training enhanced EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Enhanced EEGNet with more capacity
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(24, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv2D(48, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(96, (1, 8), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Dense
        layers.Dense(256, activation='elu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='elu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    model.fit(
        X_train_c, y_train_c,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks
    )
    
    results['EEGNet'] = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    print(f"    EEGNet: {results['EEGNet']:.1%}")
    
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 6. STACKING ENSEMBLE
# ============================================================================
print("\n[6] Creating stacking ensemble...")

estimators = [
    ('svm', SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('lr', LogisticRegression(max_iter=2000, random_state=42)),
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train_s, y_train)
results['Stacking'] = accuracy_score(y_test, stacking.predict(X_test_s))

# Voting ensemble
ensemble_estimators = [
    ('svm', svm), 
    ('rf', rf), 
    ('et', et),
    ('gb', gb),
    ('mlp', mlp)
]

ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
results['Voting'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[7] RESULTS - ULTIMATE VERSION")
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
    ('SVM', svm), 
    ('RF', rf), 
    ('ET', et), 
    ('GB', gb)
]
if 'xgb_clf' in dir():
    cv_clfs.append(('XGBoost', xgb_clf))
if 'lgb_clf' in dir():
    cv_clfs.append(('LightGBM', lgb_clf))
if 'cat_clf' in dir():
    cv_clfs.append(('CatBoost', cat_clf))

for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
