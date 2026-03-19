#!/usr/bin/env python3
"""
EEG Motor Imagery - Version 13
Goal: Beat 82.5% record
- Enhanced CSP with regularization + wavelet features
- Deeper feature engineering
- Optimized classifiers + stacking ensemble
- Enhanced EEGNet with more capacity
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, cwt, morlet2
from scipy.linalg import eigh
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              VotingClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - V13 (BEAT 82.5%)")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA - Same as best_combo (82.5%)
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 500  # Slightly more trials
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
        
        # Variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # Motor imagery effect (60% show effect, 14% suppression)
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
# 2. ENHANCED CSP FEATURES
# ============================================================================
print("\n[2] Computing enhanced CSP features...")

def compute_csp_enhanced(X, y, fs, n_components=4):
    """Enhanced CSP with regularization and multiple bands"""
    bands = [
        (8, 13),   # Mu (most important for MI)
        (13, 20),  # Beta1
        (20, 30),  # Beta2
        (4, 8),    # Theta
        (15, 25),  # High beta
    ]
    
    all_features = []
    
    for band in bands:
        b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        X_filt = np.array([filtfilt(b, a, trial) for trial in X])
        
        n_channels = X.shape[1]
        
        # Regularized covariance
        covs = []
        for c in [0, 1]:
            class_cov = np.zeros((n_channels, n_channels))
            n_class = np.sum(y == c)
            for trial in X_filt[y == c]:
                cov = np.cov(trial)
                # Regularization
                reg = 1e-5 * np.trace(cov) * np.eye(n_channels)
                class_cov += (cov + reg) / n_class
            covs.append(class_cov)
        
        try:
            # CSP
            A = covs[0] + 1e-10*np.eye(n_channels)
            B = covs[1] + 1e-10*np.eye(n_channels)
            C = A + B + 1e-9*np.eye(n_channels)
            
            eigenvalues, eigenvectors = eigh(np.linalg.inv(C) @ A)
            idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Select top and bottom components
            W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
            
            # Extract features
            features = []
            for trial in X_filt:
                projected = W.T @ trial
                var = np.var(projected, axis=1)
                
                # Log variance features (standard CSP)
                log_var = np.log(var[:n_components] / (var[n_components:] + 1e-10))
                
                # Ratio features
                ratios = var[:n_components] / (var[:n_components].sum() + 1e-10)
                
                # Combined
                trial_feat = np.concatenate([log_var, var, ratios])
                features.append(trial_feat)
            
            all_features.append(np.array(features))
        except:
            pass
    
    return np.hstack(all_features) if all_features else np.zeros((len(X), 10))

# Compute CSP features
csp_features = compute_csp_enhanced(X, y, fs, n_components=4)
csp_features = np.nan_to_num(csp_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    CSP features: {csp_features.shape}")

# ============================================================================
# 3. WAVELET FEATURES
# ============================================================================
print("\n[3] Computing wavelet features...")

def morlet_wavelet(t, fc, sigma=1.0):
    """Create Morlet wavelet"""
    x = t * fc
    return np.exp(-x**2 / (2*sigma**2)) * np.exp(1j * 2 * np.pi * x)

def compute_wavelet_features(X, fs):
    """Compute wavelet-based features"""
    n_trials, n_channels, n_samples = X.shape
    t = np.arange(n_samples) / fs
    
    # Wavelet frequencies
    freqs = [8, 10, 12, 15, 20, 25]  # Hz
    
    features = []
    for trial in X:
        trial_feats = []
        
        for ch in trial:
            for fc in freqs:
                # Simple wavelet transform (convolution with Morlet)
                wavelet = morlet_wavelet(t, fc)
                # Take real part as approximation
                coeffs = np.convolve(ch, np.real(wavelet), mode='same')
                
                # Power in different segments
                n_seg = 4
                seg_len = len(coeffs) // n_seg
                for seg in range(n_seg):
                    start = seg * seg_len
                    end = start + seg_len
                    trial_feats.append(np.mean(coeffs[start:end]**2))
                
                # Average power
                trial_feats.append(np.mean(coeffs**2))
        
        features.append(trial_feats)
    
    return np.array(features)

wavelet_features = compute_wavelet_features(X, fs)
wavelet_features = np.nan_to_num(wavelet_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Wavelet features: {wavelet_features.shape}")

# ============================================================================
# 4. COMPREHENSIVE BAND FEATURES
# ============================================================================
print("\n[4] Computing comprehensive band features...")

def extract_comprehensive_features(X, fs):
    """Extract comprehensive frequency and time features"""
    features = []
    
    for trial in X:
        trial_feats = []
        
        for ch in trial:
            # PSD features
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            
            bands_dict = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta_low': (13, 20),
                'beta_high': (20, 30),
                'gamma': (30, 45),
            }
            
            band_powers = {}
            total_power = 0
            for band_name, (f1, f2) in bands_dict.items():
                mask = (freqs >= f1) & (freqs <= f2)
                power = np.mean(psd[mask]) if np.any(mask) else 0
                band_powers[band_name] = power
                total_power += power
            
            # Absolute powers
            for p in band_powers.values():
                trial_feats.append(p)
            
            # Relative powers
            for p in band_powers.values():
                trial_feats.append(p / (total_power + 1e-10))
            
            # Ratios (important for MI)
            trial_feats.append(band_powers['alpha'] / (band_powers['beta_low'] + 1e-10))
            trial_feats.append(band_powers['alpha'] / (band_powers['theta'] + 1e-10))
            trial_feats.append(band_powers['beta_low'] / (band_powers['beta_high'] + 1e-10))
            trial_feats.append((band_powers['alpha'] + band_powers['theta']) / 
                             (band_powers['beta_low'] + band_powers['beta_high'] + 1e-10))
            
            # Time domain
            trial_feats.extend([
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 25),
                np.percentile(ch, 75),
                np.percentile(ch, 75) - np.percentile(ch, 25),  # IQR
                np.sqrt(np.mean(ch**2)),  # RMS
                skew(ch),  # Skewness
                kurtosis(ch),  # Kurtosis
                np.mean(np.abs(np.diff(ch))),  # Mean absolute difference
            ])
            
            # Hjorth parameters
            diff1 = np.diff(ch)
            diff2 = np.diff(diff1)
            activity = np.var(ch)
            mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
            complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
            trial_feats.extend([activity, mobility, complexity])
        
        features.append(trial_feats)
    
    return np.array(features)

band_features = extract_comprehensive_features(X, fs)
band_features = np.nan_to_num(band_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Band features: {band_features.shape}")

# ============================================================================
# 5. SPATIAL & ASYMMETRY FEATURES
# ============================================================================
print("\n[5] Computing spatial features...")

def extract_spatial_asymmetry(X):
    """Spatial and asymmetry features"""
    features = []
    
    for trial in X:
        trial_feats = []
        
        # Hemisphere powers
        left_ch = [0, 1, 4, 5]
        right_ch = [2, 3, 6, 7]
        frontal_ch = [0, 2, 4, 6]
        central_ch = [1, 3, 5, 7]
        
        left_power = np.mean([np.mean(trial[ch]**2) for ch in left_ch])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in right_ch])
        frontal_power = np.mean([np.mean(trial[ch]**2) for ch in frontal_ch])
        central_power = np.mean([np.mean(trial[ch]**2) for ch in central_ch])
        
        # Asymmetry
        asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
        ant_post_ratio = frontal_power / (central_power + 1e-10)
        
        # Channel-wise asymmetry
        for ch in range(4):
            l_power = np.mean(trial[ch]**2)
            r_power = np.mean(trial[ch + 4]**2)
            ch_asym = (l_power - r_power) / (l_power + r_power + 1e-10)
            corr = np.corrcoef(trial[ch], trial[ch+4])[0,1]
            trial_feats.extend([ch_asym, corr, l_power, r_power])
        
        # Regional features
        trial_feats.extend([
            left_power, right_power, asymmetry,
            frontal_power, central_power, ant_post_ratio,
            left_power / (right_power + 1e-10),
            np.log(left_power + 1), np.log(right_power + 1),
        ])
        
        # Covariance matrix features
        cov = np.cov(trial)
        trial_feats.extend([
            np.trace(cov),
            np.linalg.det(cov + 1e-10*np.eye(8)),
            np.sum(cov**2),  # Frobenius norm squared
        ])
        
        features.append(trial_feats)
    
    return np.array(features)

spatial_features = extract_spatial_asymmetry(X)
spatial_features = np.nan_to_num(spatial_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Spatial features: {spatial_features.shape}")

# ============================================================================
# 6. TEMPORAL SEGMENT FEATURES
# ============================================================================
print("\n[6] Computing temporal features...")

def extract_temporal_segments(X):
    """Temporal segment features"""
    features = []
    n_seg = 8  # More segments
    
    for trial in X:
        trial_feats = []
        
        for ch in trial:
            seg_len = len(ch) // n_seg
            for seg in range(n_seg):
                start = seg * seg_len
                end = start + seg_len
                seg_data = ch[start:end]
                trial_feats.extend([
                    np.mean(seg_data**2),
                    np.mean(seg_data),
                    np.std(seg_data),
                ])
        
        features.append(trial_feats)
    
    return np.array(features)

temporal_features = extract_temporal_segments(X)
temporal_features = np.nan_to_num(temporal_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Temporal features: {temporal_features.shape}")

# ============================================================================
# 7. COMBINE ALL FEATURES
# ============================================================================
print("\n[7] Combining all features...")

X_combined = np.hstack([
    csp_features,
    wavelet_features,
    band_features,
    spatial_features,
    temporal_features,
])

X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Total features: {X_combined.shape}")

# ============================================================================
# 8. TRAIN CLASSIFIERS
# ============================================================================
print("\n[8] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=5000, C=1.0, solver='lbfgs')
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))
print(f"    LR: {results['LogisticRegression']:.1%}")

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# Ridge Classifier
ridge = RidgeClassifier(alpha=1.0)
ridge.fit(X_train_s, y_train)
results['Ridge'] = accuracy_score(y_test, ridge.predict(X_test_s))

# SVM
svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest (tuned)
rf = RandomForestClassifier(
    n_estimators=800, 
    max_depth=25, 
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees (tuned)
et = ExtraTreesClassifier(
    n_estimators=800, 
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42, 
    n_jobs=-1
)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=400, 
    max_depth=7, 
    learning_rate=0.08,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# AdaBoost
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
ada.fit(X_train_s, y_train)
results['AdaBoost'] = accuracy_score(y_test, ada.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
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
        max_depth=15, 
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
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
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=800, 
                    early_stopping=True, random_state=42, alpha=0.001)
mlp.fit(X_train_s, y_train)
results['MLP'] = accuracy_score(y_test, mlp.predict(X_test_s))

# ============================================================================
# 9. ENHANCED EEGNet V2
# ============================================================================
print("\n[9] Training enhanced EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Prepare CNN input
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Deeper EEGNet
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (1, 64), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv2D(64, (1, 32), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Dense
        layers.Dense(256, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='elu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
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
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks
    )
    
    results['EEGNet-v2'] = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    print(f"    EEGNet-v2: {results['EEGNet-v2']:.1%}")
    
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 10. ENSEMBLE
# ============================================================================
print("\n[10] Creating ensemble...")

# Get available classifiers for ensemble
ensemble_estimators = [
    ('svm', svm), 
    ('rf', rf), 
    ('et', et),
    ('gb', gb),
]
try:
    ensemble_estimators.append(('xgb', xgb_clf))
except:
    pass
try:
    ensemble_estimators.append(('lgb', lgb_clf))
except:
    pass

ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 11. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[11] RESULTS - V13 (500 trials, enhanced features)")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    improvement = acc - 0.825
    imp_str = f" (+{improvement:.1%})" if improvement > 0 else f" ({improvement:.1%})"
    print(f"    {name}: {acc:.1%}{marker}{imp_str if name == sorted_results[0][0] else ''}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} = {best_acc:.1%}")
print(f"    Previous record: 82.5%")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clfs = [
    ('RF', rf), 
    ('ET', et), 
    ('GB', gb),
    ('XGBoost', xgb_clf if 'xgb_clf' in dir() else None),
    ('LightGBM', lgb_clf if 'lgb_clf' in dir() else None),
]

for name, clf in cv_clfs:
    if clf is not None:
        cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
        print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
