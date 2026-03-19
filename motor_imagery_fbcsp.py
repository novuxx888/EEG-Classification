#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - FBCSP + Enhanced
- Filter Bank CSP (FBCSP) - multiple frequency bands
- LightGBM (new classifier)
- Enhanced EEGNet
- Feature selection
- Harder synthetic data
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - FBCSP + Enhanced")
print("="*60)

# ============================================================================
# 1. CREATE HARDER SYNTHETIC DATA
# ============================================================================
print("\n[1] Creating harder synthetic motor imagery data...")

fs = 128
t = np.arange(0, 3.5, 1/fs)

n_trials = 400  # More data
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG with more complexity
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 12
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 24 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 3
        base = alpha + beta1 + beta2 + theta + delta
        
        # More realistic noise
        white_noise = np.random.randn(len(t)) * 12  # Increased
        drift = np.linspace(0, 3, len(t)) * np.random.randn() * 4  # Increased
        # More artifacts
        if np.random.rand() < 0.20:
            spike_idx = np.random.randint(0, len(t)-30)
            base[spike_idx:spike_idx+30] += np.random.randn(30) * 30
        
        # Muscle artifacts (occasional)
        if np.random.rand() < 0.08:
            muscle = np.random.randn(len(t)) * 15
            base += muscle
        
        base += white_noise + drift
        
        # Trial/channel variability (more variance)
        trial_factor = np.random.uniform(0.3, 1.8)
        ch_factor = np.random.uniform(0.6, 1.5)
        base *= trial_factor * ch_factor
        
        # HARDER motor imagery effect: only 50% show effect, 12% change (reduced!)
        suppression = 0.88 if np.random.rand() < 0.50 else 1.0
        noise_factor = np.random.uniform(-0.15, 0.15)  # Add noise to effect
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3, 6, 7]:
                base *= (0.88 + noise_factor) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [0, 1, 4, 5]:
                base *= (0.88 + noise_factor) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")
print(f"    Difficulty: 50% effect, 12% suppression, more noise")

# ============================================================================
# 2. FBCSP (Filter Bank Common Spatial Patterns)
# ============================================================================
print("\n[2] Computing FBCSP features...")

def compute_fbcsp_features(X, y, fs, n_components=2, bands=None):
    """Filter Bank CSP - multiple frequency bands"""
    if bands is None:
        bands = [(8, 13), (13, 18), (18, 24), (24, 30)]  # mu + sub-beta bands
    
    all_features = []
    
    for band in bands:
        b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        X_filt = np.array([filtfilt(b, a, trial) for trial in X])
        
        n_channels = X.shape[1]
        
        # CSP
        covs = []
        for c in [0, 1]:
            class_cov = np.zeros((n_channels, n_channels))
            for trial in X_filt[y == c]:
                class_cov += np.cov(trial)
            class_cov /= np.sum(y == c)
            covs.append(class_cov)
        
        try:
            eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
            idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
            
            band_features = []
            for trial in X_filt:
                projected = W.T @ trial
                var = np.var(projected, axis=1)
                band_features.append(np.log(var + 1e-10))
            
            all_features.append(band_features)
        except:
            pass
    
    return np.hstack(all_features)

# Multiple bands
bands = [(8, 13), (13, 18), (18, 24), (24, 30), (6, 10)]
fbcsp_features = compute_fbcsp_features(X, y, fs, n_components=2, bands=bands)
print(f"    FBCSP features: {fbcsp_features.shape}")

# Single best CSP (mu band)
def compute_csp(X, y, fs, n_components=3):
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            class_cov += np.cov(trial)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
    idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
    
    features = []
    for trial in X_filt:
        projected = W.T @ trial
        var = np.var(projected, axis=1)
        features.append(np.log(var + 1e-10))
    
    return np.array(features)

csp_features = compute_csp(X, y, fs, n_components=3)
print(f"    CSP features: {csp_features.shape}")

# ============================================================================
# 3. FREQUENCY BAND FEATURES
# ============================================================================
print("\n[3] Extracting frequency features...")

def extract_band_features(X, fs):
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
            mu_beta = np.mean(psd[(freqs >= 8) & (freqs <= 30)]) / (beta_low + beta_high + 1e-10)
            
            trial_feats.extend([
                alpha, beta_low, beta_high, theta, delta,
                alpha/total, beta_low/total, theta/total, delta/total,
                alpha_beta, mu_beta,
                np.mean(ch), np.std(ch), 
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.percentile(ch, 10), np.percentile(ch, 90),
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_asymmetry(X):
    """Left-right asymmetry features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in range(4):
            left_power = np.mean(trial[ch]**2)
            right_power = np.mean(trial[ch + 4]**2)
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            corr = np.corrcoef(trial[ch], trial[ch + 4])[0, 1]
            # Additional features
            left_var, right_var = np.var(trial[ch]), np.var(trial[ch+4])
            trial_feats.extend([asymmetry, corr, left_var/(right_var+1e-10)])
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
asym_features = extract_asymmetry(X)

X_combined = np.hstack([fbcsp_features, csp_features, band_features, asym_features])
print(f"    Combined features: {X_combined.shape}")

# Feature selection
print("    Applying feature selection...")
selector = SelectKBest(f_classif, k=min(80, X_combined.shape[1]))
X_selected = selector.fit_transform(X_combined, y)
print(f"    Selected features: {X_selected.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.3)
lr.fit(X_train_s, y_train)
results['LogisticRegression'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM
svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))

# Random Forest
rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=3, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))

# Extra Trees
et = ExtraTreesClassifier(n_estimators=500, max_depth=15, min_samples_split=3, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, random_state=42)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, 
        min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    xgb_clf.fit(X_train_s, y_train)
    results['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    XGBoost error: {e}")

# LightGBM (NEW!)
print("    Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM error: {e}")

# ============================================================================
# 5. ENHANCED EEGNet
# ============================================================================
print("\n[5] Training Enhanced EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    tf.random.set_seed(42)
    
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
        layers.Conv2D(8, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(16, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(32, (1, 8), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Classifier
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
    ]
    
    model.fit(X_train_c, y_train_c, epochs=80, batch_size=16, 
              validation_split=0.2, verbose=0, callbacks=callbacks)
    
    results['EEGNet-Enhanced'] = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
except Exception as e:
    print(f"    EEGNet error: {e}")

# ============================================================================
# 6. ENSEMBLE
# ============================================================================
print("\n[6] Creating ensemble...")

# Best classifiers ensemble
ensemble_estimators = [
    ('svm', svm), 
    ('rf', rf), 
    ('et', et), 
]

if 'xgb_clf' in dir() and xgb_clf is not None:
    ensemble_estimators.append(('xgb', xgb_clf))

if 'lgb_clf' in dir() and lgb_clf is not None:
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

# Cross-validation
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_clfs = [('SVM', svm), ('RF', rf), ('Ensemble', ensemble)]
if 'lgb_clf' in dir() and lgb_clf is not None:
    cv_clfs.append(('LightGBM', lgb_clf))
    
for name, clf in cv_clfs:
    cv_scores = cross_val_score(clf, scaler.fit_transform(X_selected), y, cv=cv)
    print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)