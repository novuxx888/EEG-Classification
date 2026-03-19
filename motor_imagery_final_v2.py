#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Final Version v2
- Proper CSP features (single-band optimal)
- XGBoost + RandomForest + SVM ensemble
- Harder synthetic data (more noise, subtler effects)
- Improved EEGNet architecture
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - FINAL V2")
print("="*60)

# ============================================================================
# 1. CREATE HARDEST SYNTHETIC DATA
# ============================================================================
print("\n[1] Creating HARDEST synthetic motor imagery data...")

fs = 128
t = np.arange(0, 3.0, 1/fs)

n_trials = 300  # More trials
n_channels = 8  # 4 left, 4 right motor regions

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Complex multi-frequency base EEG
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 10
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 24 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        mu = np.sin(2 * np.pi * 12 * t + np.random.rand()*2*np.pi) * 8
        base = alpha + beta1 + beta2 + theta + mu
        
        # Realistic noise (increased)
        white_noise = np.random.randn(len(t)) * 12
        # Slow drift
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 4
        # Muscle/eye artifacts (more frequent)
        if np.random.rand() < 0.2:
            spike_idx = np.random.randint(0, len(t)-25)
            base[spike_idx:spike_idx+25] += np.random.randn(25) * 30
        
        base += white_noise + drift
        
        # Strong trial/channel variability (MAJOR CHALLENGE)
        trial_factor = np.random.uniform(0.3, 1.7)
        ch_factor = np.random.uniform(0.6, 1.4)
        base *= trial_factor * ch_factor
        
        # VERY SUBTLE motor imagery effect
        # Only 55% of trials show any effect (was 60-65%)
        # Effect is only 10% change (was 12-15%)
        suppression = 0.90 if np.random.rand() < 0.55 else 1.0
        
        # Left hand = right motor cortex suppression (channels 2,3,6,7)
        # Right hand = left motor cortex suppression (channels 0,1,4,5)
        if label == 0:  # LEFT
            if ch in [2, 3, 6, 7]:
                base *= (0.90 + np.random.uniform(-0.10, 0.10)) * suppression
        else:  # RIGHT
            if ch in [0, 1, 4, 5]:
                base *= (0.90 + np.random.uniform(-0.10, 0.10)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. CSP FEATURES (single-band - optimal for motor imagery)
# ============================================================================
print("\n[2] Computing CSP features (single-band mu/alpha)...")

def compute_csp_features(X, y, fs, n_components=3):
    """Compute CSP features for mu rhythm (8-13 Hz) - best for motor imagery"""
    # Filter for mu band
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    # Compute spatial filters
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            class_cov += np.cov(trial)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    # CSP: generalized eigenvalues
    eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
    idx = np.argsort(np.abs(eigenvalues - 0.5))[::-1]  # Sort by distance from 0.5
    eigenvectors = eigenvectors[:, idx]
    
    # Select top and bottom eigenvectors
    W = np.hstack([eigenvectors[:, :n_components], 
                   eigenvectors[:, -n_components:]])
    
    # Extract features
    features = []
    for trial in X_filt:
        projected = W.T @ trial
        var = np.var(projected, axis=1)
        # Log transform
        features.append(np.log(var + 1e-10))
    
    return np.array(features), W

csp_features, W = compute_csp_features(X, y, fs, n_components=3)
print(f"    CSP features: {csp_features.shape}")

# Also compute beta-band CSP
def compute_beta_csp(X, y, fs, n_components=2):
    """Beta-band CSP (13-30 Hz)"""
    b, a = butter(4, [13/(fs/2), 30/(fs/2)], btype='band')
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

beta_csp = compute_beta_csp(X, y, fs, n_components=2)
print(f"    Beta CSP features: {beta_csp.shape}")

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
            
            # Band powers
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs <= 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs <= 30)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            total = alpha + beta_low + beta_high + theta + delta + 1e-10
            
            # Key ratios for motor imagery
            alpha_beta = alpha / (beta_low + beta_high + 1e-10)
            mu_beta = np.mean(psd[(freqs >= 10) & (freqs <= 15)]) / (beta_low + 1e-10)
            
            # Hemisphere asymmetry (important for motor imagery!)
            # Left channels: 0,1,4,5 | Right channels: 2,3,6,7
            
            trial_feats.extend([
                alpha, beta_low, beta_high, theta, delta,
                alpha/total, beta_low/total, theta/total,
                alpha_beta, mu_beta,
                np.mean(ch), np.std(ch), 
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
            ])
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
print(f"    Band features: {band_features.shape}")

# Hemisphere asymmetry features
def extract_asymmetry(X):
    """Extract left-right asymmetry features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in range(4):  # Compare pairs
            left_ch = trial[ch]
            right_ch = trial[ch + 4]
            
            # Power asymmetry
            left_power = np.mean(left_ch**2)
            right_power = np.mean(right_ch**2)
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            
            # Correlation
            corr = np.corrcoef(left_ch, right_ch)[0, 1]
            
            trial_feats.extend([asymmetry, corr])
        features.append(trial_feats)
    return np.array(features)

asym_features = extract_asymmetry(X)
print(f"    Asymmetry features: {asym_features.shape}")

# Combine all features
X_combined = np.hstack([csp_features, beta_csp, band_features, asym_features])
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

# Logistic Regression (tuned)
lr = LogisticRegression(random_state=42, max_iter=3000, C=0.3, penalty='l2')
lr.fit(X_train_s, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_s))
results['LogisticRegression'] = acc_lr

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
acc_lda = accuracy_score(y_test, lda.predict(X_test_s))
results['LDA'] = acc_lda

# SVM (tuned)
svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True)
svm.fit(X_train_s, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_s))
results['SVM-RBF'] = acc_svm

# Random Forest (tuned)
rf = RandomForestClassifier(n_estimators=400, max_depth=15, 
                            min_samples_split=2, min_samples_leaf=1,
                            random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_s))
results['RandomForest'] = acc_rf

# Extra Trees
et = ExtraTreesClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
acc_et = accuracy_score(y_test, et.predict(X_test_s))
results['ExtraTrees'] = acc_et

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=250, max_depth=5, 
                                 learning_rate=0.08, random_state=42)
gb.fit(X_train_s, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_s))
results['GradientBoosting'] = acc_gb

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_clf.fit(X_train_s, y_train)
    acc_xgb = accuracy_score(y_test, xgb_clf.predict(X_test_s))
    results['XGBoost'] = acc_xgb
except ImportError:
    print("    XGBoost not available, skipping...")

# ============================================================================
# 5. EEGNet (improved)
# ============================================================================
print("\n[5] Training EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Prepare data for CNN
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean(axis=(0,2), keepdims=True)) / (X_cnn.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Improved EEGNet architecture
    model = keras.Sequential([
        # Block 1: Temporal convolution
        layers.Conv2D(16, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        
        # Block 2: Spatial filtering (depthwise)
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False, 
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.3),
        
        # Block 3: Separable convolution
        layers.Conv2D(32, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(64, (1, 8), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Classifier
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.5),
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
    
    model.fit(X_train_c, y_train_c, epochs=80, batch_size=16, 
              validation_split=0.2, verbose=0, callbacks=callbacks)
    
    acc_cnn = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    results['EEGNet'] = acc_cnn
    print(f"    EEGNet accuracy: {acc_cnn:.1%}")
except Exception as e:
    print(f"    EEGNet failed: {e}")

# ============================================================================
# 6. ENSEMBLE
# ============================================================================
print("\n[6] Creating ensemble...")

# Voting ensemble
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('rf', rf),
        ('xgb', xgb_clf if 'xgb_clf' in dir() else None)
    ],
    voting='soft' if 'xgb_clf' in dir() else 'hard'
)

# Filter out None estimators
valid_estimators = [e for e in [
    ('svm', svm),
    ('rf', rf),
    ('et', et),
    ('lr', lr)
] if e[1] is not None]

ensemble = VotingClassifier(estimators=valid_estimators, voting='soft')
ensemble.fit(X_train_s, y_train)
acc_ens = accuracy_score(y_test, ensemble.predict(X_test_s))
results['Ensemble'] = acc_ens

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

# Cross-validation on best models
print("\n    Cross-validation (5-fold, stratified):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in [('SVM', svm), ('RF', rf), ('Ensemble', ensemble)]:
    if hasattr(clf, 'predict'):
        cv_scores = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=cv)
        print(f"      {name}: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("SUMMARY FOR README:")
print("="*60)
print("| Classifier | Accuracy |")
print("|------------|----------|")
for name, acc in sorted_results:
    print(f"| {name} | {acc:.0%} |")

print("\n" + "="*60)
print("DONE!")
print("="*60)
