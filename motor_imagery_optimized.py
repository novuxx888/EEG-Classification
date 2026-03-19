#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Optimized Version
- Multi-band CSP features (improved)
- RandomForest + optimized parameters
- Balanced difficulty synthetic data
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
print("EEG MOTOR IMAGERY - OPTIMIZED VERSION")
print("="*60)

# ============================================================================
# 1. CREATE BALANCED SYNTHETIC DATA (hard but solvable)
# ============================================================================
print("\n[1] Creating balanced synthetic motor imagery data...")

fs = 128
t = np.arange(0, 3.5, 1/fs)  # 3.5 seconds

n_trials = 200
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base signal
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 12
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 5
        beta2 = np.sin(2 * np.pi * 24 * t + np.random.rand()*2*np.pi) * 3
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        base = alpha + beta1 + beta2 + theta
        
        # Realistic noise
        white_noise = np.random.randn(len(t)) * 8
        drift = np.linspace(0, 1.5, len(t)) * np.random.randn() * 3
        # Occasional artifacts
        if np.random.rand() < 0.1:
            spike_idx = np.random.randint(0, len(t)-15)
            base[spike_idx:spike_idx+15] += np.random.randn(15) * 20
        
        base += white_noise + drift
        
        # Trial variability
        trial_factor = np.random.uniform(0.5, 1.5)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # Motor imagery effect - balanced (65% of trials show effect, 15% suppression)
        suppression = 0.85 if np.random.rand() < 0.65 else 1.0
        
        if label == 0:  # LEFT - right motor cortex (channels 2,3)
            if ch in [2, 3]:
                base *= (0.85 + np.random.uniform(-0.1, 0.1)) * suppression
        else:  # RIGHT - left motor cortex (channels 4,5)
            if ch in [4, 5]:
                base *= (0.85 + np.random.uniform(-0.1, 0.1)) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data shape: {X.shape}")
print(f"    Labels: Left={np.sum(y==0)}, Right={np.sum(y==1)}")

# ============================================================================
# 2. MULTI-BAND CSP FEATURES
# ============================================================================
print("\n[2] Computing multi-band CSP features...")

def compute_csp_features(X, y, fs, n_components=2):
    """Compute CSP features for multiple bands"""
    bands = [(8, 13), (13, 22), (22, 30)]  # mu, low-beta, high-beta
    all_features = []
    
    for low, high in bands:
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        X_filt = np.array([filtfilt(b, a, trial) for trial in X])
        
        # CSP computation
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
        
        # Extract features per trial
        trial_features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            # Log transform for better distribution
            trial_features.append(np.log(var + 1e-10))
        all_features.append(trial_features)
    
    return np.hstack(all_features)

csp_features = compute_csp_features(X, y, fs, n_components=2)
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
            
            # Band powers
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            total = alpha + beta + theta + delta + 1e-10
            
            # Ratios
            alpha_beta = alpha / (beta + 1e-10)
            alpha_theta = alpha / (theta + 1e-10)
            
            trial_feats.extend([
                alpha, beta, theta, delta,
                alpha/total, beta/total, theta/total,
                alpha_beta, alpha_theta,
                np.mean(ch), np.std(ch), np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),
            ])
        features.append(trial_feats)
    return np.array(features)

band_features = extract_band_features(X, fs)
print(f"    Band features: {band_features.shape}")

# Combine
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

# Logistic Regression (tuned)
lr = LogisticRegression(random_state=42, max_iter=2000, C=0.5, penalty='l2')
lr.fit(X_train_s, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_s))
results['LogisticRegression'] = acc_lr

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
acc_lda = accuracy_score(y_test, lda.predict(X_test_s))
results['LDA'] = acc_lda

# SVM (tuned)
svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
svm.fit(X_train_s, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_s))
results['SVM-RBF'] = acc_svm

# Random Forest (tuned)
rf = RandomForestClassifier(n_estimators=300, max_depth=12, 
                            min_samples_split=3, min_samples_leaf=1,
                            random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_s))
results['RandomForest'] = acc_rf

# Extra Trees
et = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train)
acc_et = accuracy_score(y_test, et.predict(X_test_s))
results['ExtraTrees'] = acc_et

# Gradient Boosting (tuned)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, 
                                 learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_s))
results['GradientBoosting'] = acc_gb

# ============================================================================
# 5. EEGNet (if available)
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
        
        layers.Flatten(),
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train_c, y_train_c, epochs=50, batch_size=16, 
              validation_split=0.2, verbose=0, callbacks=[early_stop])
    
    acc_cnn = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    results['EEGNet'] = acc_cnn
except Exception as e:
    print(f"    EEGNet error: {e}")

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
print("\n    Cross-validation (5-fold, stratified):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in [('LR', lr), ('SVM', svm), ('RF', rf), ('ExtraTrees', et)]:
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