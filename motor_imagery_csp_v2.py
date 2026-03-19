#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - CSP Optimized + RF/XGB + EEGNet v2
- Balanced harder data (55% effect, 13% suppression)
- Enhanced CSP features (mu + beta)
- Optimized RF/XGBoost
- Better EEGNet
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - CSP Optimized + RF/XGB + EEGNet v2")
print("="*60)

# ============================================================================
# 1. CREATE BALANCED HARDER SYNTHETIC DATA
# ============================================================================
print("\n[1] Creating balanced harder synthetic motor imagery data...")

fs = 128
t = np.arange(0, 3.5, 1/fs)

n_trials = 350
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 12
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        base = alpha + beta1 + beta2 + theta
        
        # Noise
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-25)
            base[spike_idx:spike_idx+25] += np.random.randn(25) * 25
        
        base += white_noise + drift
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.35, 1.65)
        ch_factor = np.random.uniform(0.65, 1.35)
        base *= trial_factor * ch_factor
        
        # BALANCED: 55% show effect, 13% suppression (vs previous 65%/15%)
        show_effect = np.random.rand() < 0.55
        suppression = 0.87 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.87 + np.random.uniform(-0.12, 0.12)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.87 + np.random.uniform(-0.12, 0.12)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: Balanced-hard (55% effect, 13% suppression)")

# ============================================================================
# 2. CSP FEATURES (optimized)
# ============================================================================
print("\n[2] Computing CSP features...")

def compute_csp(X, y, fs, n_components=3, band=(8, 13)):
    """Common Spatial Patterns"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            cov = np.cov(trial)
            class_cov += cov / np.trace(cov + 1e-10)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    try:
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1] + 1e-10*np.eye(n_channels))
        idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigenvectors = eigenvectors[:, idx]
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            features.append(np.log(var + 1e-10))
        
        return np.array(features)
    except:
        return np.zeros((len(X), n_components*2))

# CSP for different bands
csp_mu = compute_csp(X, y, fs, n_components=3, band=(8, 13))    # Mu (8-13 Hz)
csp_beta1 = compute_csp(X, y, fs, n_components=2, band=(13, 20))  # Beta1
csp_beta2 = compute_csp(X, y, fs, n_components=2, band=(20, 30))  # Beta2

print(f"    CSP (mu): {csp_mu.shape}, CSP (beta1): {csp_beta1.shape}, CSP (beta2): {csp_beta2.shape}")

# ============================================================================
# 3. ADDITIONAL FEATURES
# ============================================================================
print("\n[3] Extracting additional features...")

def extract_band_features(X, fs):
    """Frequency band power features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=64)
            
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs <= 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs <= 30)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            total = alpha + beta_low + beta_high + theta + delta + 1e-10
            
            alpha_beta = alpha / (beta_low + beta_high + 1e-10)
            
            trial_feats.extend([
                alpha, beta_low, beta_high, theta, delta,
                alpha/total, beta_low/total, beta_high/total, theta/total,
                alpha_beta,
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
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
            trial_feats.extend([asymmetry, corr, left_power, right_power])
        features.append(trial_feats)
    return np.array(features)

def extract_connectivity(X):
    """Channel connectivity features"""
    features = []
    for trial in X:
        trial_feats = []
        # Correlation matrix as features
        corr_matrix = np.corrcoef(trial)
        # Upper triangle (excluding diagonal)
        upper = corr_matrix[np.triu_indices(n_channels, k=1)]
        trial_feats.extend(upper)
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
asym_features = extract_asymmetry(X)
conn_features = extract_connectivity(X)

# Combine all features
X_combined = np.hstack([csp_mu, csp_beta1, csp_beta2, band_features, asym_features, conn_features])
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
lr = LogisticRegression(random_state=42, max_iter=2000, C=0.8)
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM (tuned)
svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest (optimized)
rf = RandomForestClassifier(
    n_estimators=400, 
    max_depth=12, 
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees
et = ExtraTreesClassifier(
    n_estimators=400, 
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42, 
    n_jobs=-1
)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# XGBoost (optimized)
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
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
        n_estimators=400, 
        max_depth=8, 
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42, 
        verbose=-1,
        n_jobs=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM error: {e}")

# ============================================================================
# 5. EEGNet v2
# ============================================================================
print("\n[5] Training EEGNet v2...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Prepare data
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # EEGNet v2 - slightly deeper
    model = keras.Sequential([
        layers.Conv2D(16, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.3),
        
        layers.Conv2D(32, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (1, 8), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(32, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    model.fit(
        X_train_c, y_train_c,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )
    
    results['EEGNet'] = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    print(f"    EEGNet: {results['EEGNet']:.1%}")
    
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 6. ENSEMBLE
# ============================================================================
print("\n[6] Creating ensemble...")

ensemble_estimators = [
    ('svm', svm), 
    ('rf', rf), 
    ('et', et),
    ('lr', lr)
]
if 'xgb_clf' in dir():
    ensemble_estimators.append(('xgb', xgb_clf))
if 'lgb_clf' in dir():
    ensemble_estimators.append(('lgb', lgb_clf))

ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[7] RESULTS")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} = {best_acc:.1%}")
print(f"    Previous best: 81%")

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clfs = [
    ('SVM', svm), 
    ('RF', rf), 
    ('ET', et), 
    ('Ensemble', ensemble)
]
if 'xgb_clf' in dir():
    cv_clfs.append(('XGBoost', xgb_clf))
if 'lgb_clf' in dir():
    cv_clfs.append(('LightGBM', lgb_clf))

cv_results = {}
for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
    cv_results[name] = cv_scores.mean()
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
