#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Enhanced Version
- XGBoost + better CSP
- Even harder synthetic data
- Multi-band features
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - ENHANCED WITH XGBOOST")
print("="*60)

# ============================================================================
# 1. CREATE HARDER SYNTHETIC DATA
# ============================================================================
print("\n[1] Creating HARDER synthetic motor imagery data...")

fs = 128
t = np.arange(0, 3, 1/fs)

n_trials = 250  # More trials
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # More complex base - multiple frequencies with phase variations
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 10
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 24 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 3
        base = alpha + beta1 + beta2 + theta
        
        # More realistic noise
        white_noise = np.random.randn(len(t)) * 10
        # Drift (very realistic)
        drift = np.linspace(0, 2, len(t)) * np.random.randn() * 3
        # Muscle artifacts (occasional)
        if np.random.rand() < 0.15:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        # Cross-trial and cross-channel variability (MAJOR challenge!)
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # SUBTLE motor imagery effect (even harder!)
        # 60% of trials show effect (was 70%)
        # Effect is even weaker (12% instead of 15%)
        suppression = 0.88 if np.random.rand() < 0.6 else 1.0
        
        if label == 0:  # LEFT - right motor cortex
            if ch in [2, 3]:
                base *= (0.88 + np.random.uniform(-0.12, 0.12)) * suppression
        else:  # RIGHT - left motor cortex
            if ch in [4, 5]:
                base *= (0.88 + np.random.uniform(-0.12, 0.12)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. CSP FEATURES (improved - multi-band)
# ============================================================================
print("\n[2] Computing CSP features (multi-band)...")

def compute_csp_multi(X, y, fs, n_components=2):
    """Multi-band CSP - extract features from multiple frequency bands"""
    bands = [(8, 13), (13, 22), (22, 30)]  # mu, low-beta, high-beta
    all_features = []
    
    for low, high in bands:
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        X_filt = np.array([filtfilt(b, a, trial) for trial in X])
        
        # CSP
        covs = []
        for c in [0, 1]:
            class_cov = np.zeros((n_channels, n_channels))
            for trial in X_filt[y == c]:
                class_cov += np.cov(trial)
            class_cov /= np.sum(y == c)
            covs.append(class_cov)
        
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        W = np.hstack([eigenvectors[:, :n_components], 
                       eigenvectors[:, -n_components:]])
        
        # Features (for each trial)
        trial_features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            trial_features.append(np.log(var + 1e-10))
        all_features.append(trial_features)
    
    # Stack features from all bands
    return np.hstack(all_features)

def extract_csp_features(X, W, fs):
    """Extract CSP features from pre-computed W"""
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    features = []
    for trial in X_filt:
        projected = W.T @ trial
        var = np.var(projected, axis=1)
        features.append(np.log(var + 1e-10))
    return np.array(features)

# Multi-band CSP
csp_features = compute_csp_multi(X, y, fs, n_components=2)
print(f"    CSP features (multi-band): {csp_features.shape}")

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
            beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            total = alpha + beta + theta + delta + 1e-10
            
            # Ratios (important for motor imagery)
            alpha_beta = alpha / (beta + 1e-10)
            alpha_theta = alpha / (theta + 1e-10)
            
            trial_feats.extend([
                alpha, beta, theta, delta,
                alpha/total, beta/total, theta/total,
                alpha_beta, alpha_theta,
                np.mean(ch), np.std(ch), np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.percentile(ch, 10), np.percentile(ch, 90)
            ])
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
print(f"    Band features: {band_features.shape}")

# Combine all features
X_combined = np.hstack([csp_features, band_features])
print(f"    Combined: {X_combined.shape}")

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
lr = LogisticRegression(random_state=42, max_iter=2000, C=0.1)
lr.fit(X_train_s, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_s))
results['LogisticRegression'] = acc_lr

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
acc_lda = accuracy_score(y_test, lda.predict(X_test_s))
results['LDA'] = acc_lda

# SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_s, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_s))
results['SVM-RBF'] = acc_svm

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, 
                            min_samples_split=5, random_state=42)
rf.fit(X_train_s, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_s))
results['RandomForest'] = acc_rf

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                 learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_s))
results['GradientBoosting'] = acc_gb

# Extra Randomized Trees (more diversity)
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42)
et.fit(X_train_s, y_train)
acc_et = accuracy_score(y_test, et.predict(X_test_s))
results['ExtraTrees'] = acc_et

# ============================================================================
# 5. EEGNet
# ============================================================================
print("\n[5] Training EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean()) / (X_cnn.std() + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Improved EEGNet with more regularization
    model = keras.Sequential([
        layers.Conv2D(8, (1, 16), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        
        layers.Conv2D(16, (1, 8), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (1, 4), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(32, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_c, y_train_c, epochs=50, batch_size=16, 
              validation_split=0.2, verbose=0, callbacks=[early_stop])
    
    acc_cnn = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    results['EEGNet'] = acc_cnn
except Exception as e:
    print(f"    EEGNet failed: {e}")

# ============================================================================
# 6. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[6] RESULTS")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    Best: {best_name} = {best_acc:.1%}")

# Cross-validation
print("\n    Cross-validation scores (5-fold):")
for name, clf in [('LR', lr), ('SVM', svm), ('RF', rf), ('GB', gb)]:
    cv = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=5)
    print(f"      {name}: {cv.mean():.1%} ± {cv.std():.1%}")

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