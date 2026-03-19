#!/usr/bin/env python3
"""
EEG Motor Imagery - NEW APPROACH v4b
1. CSP features + 6-band FBCSP
2. RandomForest + XGBoost + GradientBoosting ensemble
3. Harder synthetic data (55% effect, 12% suppression)
4. EEGNet with time-freq representation
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, spectrogram
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
print("EEG MOTOR IMAGERY - NEW APPROACH v4b")
print("="*60)

# ============================================================================
# 1. CREATE HARDER SYNTHETIC DATA
# ============================================================================
print("\n[1] Creating harder synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 500  # More trials for harder data
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency EEG with more realistic dynamics
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 2
        
        base = alpha + beta1 + beta2 + theta + delta
        
        # Noise - more realistic
        white_noise = np.random.randn(len(t)) * 12
        drift = np.linspace(0, 3, len(t)) * np.random.randn() * 4
        # Add occasional artifacts
        if np.random.rand() < 0.15:
            spike_idx = np.random.randint(0, len(t)-30)
            base[spike_idx:spike_idx+30] += np.random.randn(30) * 30
        # Add electrode drift
        if np.random.rand() < 0.1:
            base += np.sin(np.linspace(0, np.random.rand()*2*np.pi, len(t))) * 5
        
        base += white_noise + drift
        
        # Trial/channel variability - more variance
        trial_factor = np.random.uniform(0.35, 1.65)
        ch_factor = np.random.uniform(0.65, 1.35)
        base *= trial_factor * ch_factor
        
        # 55% show effect, 12% suppression (HARDER than 65%/16%)
        show_effect = np.random.rand() < 0.55
        suppression = 0.88 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.88 + np.random.uniform(-0.10, 0.10)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.88 + np.random.uniform(-0.10, 0.10)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: 55% effect, 12% suppression (HARDER)")

# ============================================================================
# 2. CSP FEATURES + 6-BAND FBCSP
# ============================================================================
print("\n[2] Computing CSP + FBCSP features...")

def compute_csp(X, y, fs, n_components=4, band=(8, 13)):
    """CSP with regularization"""
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

# 6-band FBCSP (delta, theta, mu, low-mu, beta1, beta2)
bands = [
    (2, 4),    # Delta
    (4, 8),    # Theta
    (8, 13),   # Mu
    (6, 12),   # Low-mu
    (13, 20),  # Beta1
    (20, 30),  # Beta2
]

def compute_fbcsp_features(X, y, fs, bands):
    """Filter Bank CSP features"""
    all_features = []
    
    for band in bands:
        csp_feats = compute_csp(X, y, fs, n_components=4, band=band)
        all_features.append(csp_feats)
    
    return np.hstack(all_features)

csp_features = compute_fbcsp_features(X, y, fs, bands)
print(f"    CSP features: {csp_features.shape}")

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Computing additional features...")

def extract_features(X, fs):
    """Extract frequency, time, and spatial features"""
    features = []
    
    for trial in X:
        trial_features = []
        
        # Frequency band powers
        for ch in trial:
            # Band powers using Welch
            freqs, psd = welch(ch, fs, nperseg=128)
            
            # Band powers
            def band_power(fmin, fmax):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                return np.mean(psd[idx])
            
            trial_features.append(band_power(2, 4))   # Delta
            trial_features.append(band_power(4, 8))   # Theta
            trial_features.append(band_power(8, 13))  # Alpha/Mu
            trial_features.append(band_power(13, 20)) # Beta1
            trial_features.append(band_power(20, 30))  # Beta2
            
            # Band power ratios (important for motor imagery)
            alpha = band_power(8, 13)
            beta = band_power(13, 30)
            theta = band_power(4, 8)
            trial_features.append(alpha / (beta + 1e-10))
            trial_features.append(alpha / (theta + 1e-10))
            trial_features.append(beta / (theta + 1e-10))
            
            # Time domain
            trial_features.append(np.mean(ch))
            trial_features.append(np.std(ch))
            trial_features.append(np.max(np.abs(ch)))
            trial_features.append(np.percentile(ch, 75) - np.percentile(ch, 25))
            trial_features.append(np.sqrt(np.mean(ch**2)))  # RMS
        
        # Hemisphere asymmetry features
        left_hemi = np.array([trial[:4]])
        right_hemi = np.array([trial[4:]])
        
        for freq in [8, 10, 12, 18, 22]:
            # Compute band power for asymmetry
            for bmin, bmax in [(freq-2, freq+2)]:
                b, a = butter(4, [bmin/(fs/2), bmax/(fs/2)], btype='band')
                left_filt = filtfilt(b, a, left_hemi.mean(axis=0))
                right_filt = filtfilt(b, a, right_hemi.mean(axis=0))
                trial_features.append(np.mean(left_filt**2) - np.mean(right_filt**2))
        
        # Temporal segment features
        n_segments = 8
        segment_size = len(t) // n_segments
        for seg in range(n_segments):
            seg_data = trial[:, seg*segment_size:(seg+1)*segment_size]
            trial_features.append(np.mean(seg_data))
            trial_features.append(np.std(seg_data))
        
        features.append(trial_features)
    
    return np.array(features)

freq_features = extract_features(X, fs)
print(f"    Frequency + time features: {freq_features.shape}")

# Combine all features
X_features = np.hstack([csp_features, freq_features])
print(f"    Combined features: {X_features.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# RandomForest
print("    Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=800, max_depth=20, min_samples_split=3,
    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))
print(f"    RandomForest: {results['RandomForest']:.1%}")

# ExtraTrees
print("    Training ExtraTrees...")
et = ExtraTreesClassifier(
    n_estimators=800, max_depth=20, min_samples_split=3,
    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))
print(f"    ExtraTrees: {results['ExtraTrees']:.1%}")

# GradientBoosting
print("    Training GradientBoosting...")
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
print("    Training MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64), max_iter=800,
    early_stopping=True, validation_fraction=0.15, random_state=42
)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))
print(f"    MLP: {results['MLP']:.1%}")

# ============================================================================
# 5. EEGNet with Time-Frequency
# ============================================================================
print("\n[5] Training EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Prepare data for CNN
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # EEGNet architecture
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
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train_c, y_train_c,
        epochs=80, batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    
    y_pred_cnn = (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten()
    results['EEGNet'] = accuracy_score(y_test_c, y_pred_cnn)
    print(f"    EEGNet: {results['EEGNet']:.1%}")
    
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================
print("\n[6] Cross-validation (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('RF', rf), ('ET', et), ('GB', gb)]:
    cv_scores = cross_val_score(clf, X_features, y, cv=cv, scoring='accuracy')
    print(f"    {name} CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "="*60)
print("RESULTS (Harder data: 55% effect, 12% suppression)")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_results:
    print(f"    {name}: {acc:.1%}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} at {best_acc:.1%}")

# Compare to previous
print("\n" + "="*60)
print("COMPARISON TO PREVIOUS:")
print("    Previous best (easier data): 88.9%")
print(f"    Current best (harder data): {best_acc:.1%}")
print("="*60)

# Save results
with open('/Users/lobter/.openclaw/workspace/EEG-Classification/results_v4b.txt', 'w') as f:
    f.write(f"New Approach v4b Results (Harder data: 55% effect, 12% suppression)\n")
    f.write("="*50 + "\n")
    for name, acc in sorted_results:
        f.write(f"{name}: {acc:.2%}\n")
    f.write(f"\nBest: {best_name} at {best_acc:.2%}\n")
    f.write(f"\nNote: Harder data than previous (55%/12% vs 65%/16%)\n")

print("\nDone!")
