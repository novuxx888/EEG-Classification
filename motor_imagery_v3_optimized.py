#!/usr/bin/env python3
"""
EEG Motor Imagery - OPTIMIZED VERSION v3
1. Enhanced CSP features with regularization
2. Optimized RF/XGBoost with better tuning
3. Balanced difficulty (65% effect, 16% suppression)
4. Improved feature engineering
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
print("EEG MOTOR IMAGERY - OPTIMIZED v3")
print("="*60)

# ============================================================================
# 1. CREATE BALANCED SYNTHETIC DATA (slightly harder than best)
# ============================================================================
print("\n[1] Creating balanced synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 450  # More trials
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
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # 65% show effect, 16% suppression (slightly harder than 60%/14%)
        show_effect = np.random.rand() < 0.65
        suppression = 0.84 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.84 + np.random.uniform(-0.08, 0.08)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.84 + np.random.uniform(-0.08, 0.08)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: 65% effect, 16% suppression")

# ============================================================================
# 2. ENHANCED CSP + FBCSP FEATURES
# ============================================================================
print("\n[2] Computing Enhanced CSP + FBCSP features...")

def compute_enhanced_csp(X, y, fs, n_components=4, band=(8, 13)):
    """Enhanced CSP with regularized covariance"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    def compute_cov(trial):
        cov = np.cov(trial)
        reg = 1e-5 * np.trace(cov)
        return cov + reg * np.eye(n_channels)
    
    class_covs = {}
    for c in [0, 1]:
        class_trials = X_filt[y == c]
        n_class = len(class_trials)
        avg_cov = np.zeros((n_channels, n_channels))
        for trial in class_trials:
            cov = compute_cov(trial)
            avg_cov += cov / n_class
        class_covs[c] = avg_cov
    
    try:
        cov_sum = class_covs[0] + class_covs[1]
        cov_sum += 1e-6 * np.trace(cov_sum) * np.eye(n_channels)
        
        eigenvalues, eigenvectors = eigh(np.linalg.inv(cov_sum) @ class_covs[0])
        sorted_idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        W = np.hstack([
            eigenvectors[:, :n_components],
            eigenvectors[:, -n_components:]
        ])
        
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            variances = np.var(projected, axis=1)
            log_var = np.log(variances[:n_components] / (variances[n_components:] + 1e-10))
            features.append(np.concatenate([log_var, variances]))
        
        return np.array(features)
    except:
        return np.zeros((len(X), n_components * 2))

# 6-band FBCSP
bands = [
    (2, 4),    # Delta
    (4, 8),    # Theta
    (8, 13),   # Mu
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (6, 12),   # Low mu
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_enhanced_csp(X, y, fs, n_components=4, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
print(f"    FBCSP total: {fbcsp.shape}")
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Extracting additional features...")

def extract_band_features(X, fs):
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            
            delta = np.mean(psd[(freqs >= 1) & (freqs < 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs < 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs < 30)])
            total = delta + theta + alpha + beta_low + beta_high + 1e-10
            
            trial_feats.extend([delta, theta, alpha, beta_low, beta_high])
            trial_feats.extend([
                delta/total, theta/total, alpha/total, 
                beta_low/total, beta_high/total
            ])
            trial_feats.extend([
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta,
                (alpha + theta)/(beta_low + beta_high + 1e-10),
                theta/alpha,
                beta_low/beta_high
            ])
            trial_feats.extend([
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2)),
                np.median(np.abs(ch - np.median(ch))),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_asymmetry(X):
    features = []
    for trial in X:
        left_channels = [0, 1, 4, 5]
        right_channels = [2, 3, 6, 7]
        
        left_power = np.mean([np.mean(trial[ch]**2) for ch in left_channels])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in right_channels])
        
        asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
        
        left_alpha = np.mean([np.mean(trial[ch]**2) for ch in left_channels])
        right_alpha = np.mean([np.mean(trial[ch]**2) for ch in right_channels])
        alpha_asym = (left_alpha - right_alpha) / (left_alpha + right_alpha + 1e-10)
        
        corrs = []
        for i, (lc, rc) in enumerate(zip(left_channels, right_channels)):
            corr = np.corrcoef(trial[lc], trial[rc])[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0)
        
        features.append([
            asymmetry, alpha_asym,
            np.log(left_power + 1), np.log(right_power + 1),
            left_power / (right_power + 1e-10),
            np.mean(corrs), np.std(corrs)
        ])
    return np.array(features)

def extract_spatial_features(X):
    features = []
    for trial in X:
        frontal = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 4, 6]])
        central = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        left = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        
        channel_powers = [np.mean(trial[ch]**2) for ch in range(n_channels)]
        
        features.append([
            frontal, central, left, right,
            frontal / (central + 1e-10),
            left / (right + 1e-10),
            (left - right) / (left + right + 1e-10),
            (frontal - central) / (frontal + central + 1e-10),
            np.max(channel_powers), np.min(channel_powers),
            np.argmax(channel_powers), np.argmin(channel_powers),
            np.max(channel_powers) / (np.min(channel_powers) + 1e-10)
        ])
    return np.array(features)

def extract_temporal_features(X):
    features = []
    n_seg = 6
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
    features = []
    for trial in X:
        trial_feats = []
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                corr = np.corrcoef(trial[i], trial[j])[0, 1]
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
lr = LogisticRegression(random_state=42, max_iter=5000, C=0.5, solver='lbfgs')
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))
print(f"    LR: {results['LogisticRegression']:.1%}")

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))
print(f"    LDA: {results['LDA']:.1%}")

# SVM
svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))
print(f"    SVM-RBF: {results['SVM-RBF']:.1%}")

# RandomForest
rf = RandomForestClassifier(
    n_estimators=800, max_depth=20, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))
print(f"    RandomForest: {results['RandomForest']:.1%}")

# ExtraTrees
et = ExtraTreesClassifier(
    n_estimators=800, max_depth=20, min_samples_split=2,
    min_samples_leaf=1, random_state=42, n_jobs=-1
)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))
print(f"    ExtraTrees: {results['ExtraTrees']:.1%}")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))
print(f"    GradientBoosting: {results['GradientBoosting']:.1%}")

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.15, reg_lambda=1.2,
        random_state=42, use_label_encoder=False,
        eval_metric='logloss', verbosity=0, n_jobs=-1
    )
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
    print(f"    XGBoost: {results['XGBoost']:.1%}")
except Exception as e:
    print(f"    XGBoost error: {e}")

# LightGBM
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=700, max_depth=12, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.15, reg_lambda=1.2,
        random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
    print(f"    LightGBM: {results['LightGBM']:.1%}")
except Exception as e:
    print(f"    LightGBM error: {e}")

# MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64), max_iter=800,
    early_stopping=True, validation_fraction=0.15, random_state=42
)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))
print(f"    MLP: {results['MLP']:.1%}")

# ============================================================================
# 5. EEGNet
# ============================================================================
print("\n[5] Training EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    model = keras.Sequential([
        layers.Conv2D(8, (1, 64), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        
        layers.Conv2D(16, (1, 32), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='elu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    model.fit(X_train_c, y_train_c, epochs=100, batch_size=16,
              validation_split=0.2, verbose=0, callbacks=callbacks)
    
    results['EEGNet'] = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    print(f"    EEGNet: {results['EEGNet']:.1%}")
    
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 6. ENSEMBLE
# ============================================================================
print("\n[6] Creating ensemble...")

ensemble_estimators = [
    ('svm', svm), ('rf', rf), ('et', et), ('lr', lr), ('gb', gb), ('mlp', mlp)
]
if 'xgb_clf' in dir():
    ensemble_estimators.append(('xgb', xgb_clf))
if 'lgb_clf' in dir():
    ensemble_estimators.append(('lgb', lgb_clf))

ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))
print(f"    Ensemble: {results['Ensemble']:.1%}")

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[7] RESULTS - BALANCED DATA (65% effect, 16% suppression)")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    BEST: {best_name} = {best_acc:.1%}")
print(f"    Previous best: 82.5%")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clfs = [('RF', rf), ('ET', et), ('GB', gb)]
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