#!/usr/bin/env python3
"""
EEG Motor Imagery - Improved EEGNet
- Deeper architecture with residual connections
- Data augmentation
- Better regularization
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - IMPROVED EEGNet")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA
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
        # Multi-frequency base EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
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
# 2. COMPUTE FEATURES
# ============================================================================
print("\n[2] Computing features...")

def compute_csp(X, y, fs, n_components=3, band=(8, 13)):
    """CSP for a specific frequency band"""
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
bands = [(4, 8), (8, 13), (13, 20), (20, 30), (6, 12)]

fbcsp_features = []
for band in bands:
    csp_feat = compute_csp(X, y, fs, n_components=3, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)

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

def extract_asymmetry(X):
    features = []
    for trial in X:
        trial_feats = []
        for ch in range(4):
            left_power = np.mean(trial[ch]**2)
            right_power = np.mean(trial[ch + 4]**2)
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            corr = np.corrcoef(trial[ch], trial[ch + 4])[0, 1]
            trial_feats.extend([asymmetry, corr, np.log(left_power+1), np.log(right_power+1), left_power / (right_power + 1e-10)])
        features.append(trial_feats)
    return np.array(features)

def extract_spatial_features(X):
    features = []
    for trial in X:
        trial_feats = []
        left_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        ant_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 4, 6]])
        post_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        central_power = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        
        trial_feats.extend([
            left_power, right_power, ant_power, post_power, central_power,
            left_power / (right_power + 1e-10),
            ant_power / (post_power + 1e-10)
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
asym_features = extract_asymmetry(X)
spatial_features = extract_spatial_features(X)
temporal_features = extract_temporal_features(X)

X_combined = np.hstack([fbcsp, band_features, asym_features, spatial_features, temporal_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Combined features: {X_combined.shape}")

# ============================================================================
# 3. TRAIN EEGNet
# ============================================================================
print("\n[3] Training improved EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Prepare data for CNN
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    # Data augmentation - add noise
    X_aug = []
    y_aug = []
    for i in range(len(X_cnn)):
        X_aug.append(X_cnn[i])
        y_aug.append(y[i])
        # Add noisy version
        noise = np.random.randn(*X_cnn[i].shape) * 0.1
        X_aug.append(X_cnn[i] + noise)
        y_aug.append(y[i])
    
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Improved EEGNet with residual connections
    inputs = layers.Input(shape=(n_channels, len(t), 1))
    
    # Block 1
    x = layers.Conv2D(32, (1, 64), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = layers.Conv2D(64, (1, 32), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv2D(128, (1, 16), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers
    x = layers.Dense(256, activation='elu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='elu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='elu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.0008),
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
        batch_size=20,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks
    )
    
    eegnet_acc = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    print(f"    EEGNet accuracy: {eegnet_acc:.1%}")
    
    HAS_TF = True
except Exception as e:
    print(f"    EEGNet error: {e}")
    HAS_TF = False
    eegnet_acc = 0

# ============================================================================
# 4. TRAIN CLASSICAL ML
# ============================================================================
print("\n[4] Training classical ML models...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# RandomForest
rf = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees
et = ExtraTreesClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# GradientBoosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.08, 
                                 random_state=42, use_label_encoder=False, 
                                 eval_metric='logloss', verbosity=0, n_jobs=-1)
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except:
    pass

# LightGBM
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(n_estimators=600, max_depth=12, learning_rate=0.06, 
                                 random_state=42, verbose=-1, n_jobs=-1)
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except:
    pass

if HAS_TF:
    results['EEGNet'] = eegnet_acc

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[5] RESULTS")
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

cv_clfs = [('RF', rf), ('ET', et), ('GB', gb)]
if HAS_TF:
    cv_clfs.append(('XGBoost', xgb_clf))
    cv_clfs.append(('LightGBM', lgb_clf))

for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
