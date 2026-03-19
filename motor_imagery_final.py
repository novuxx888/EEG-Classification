#!/usr/bin/env python3
"""
EEG Motor Imagery - Final Enhanced Version
CSP features, Multiple classifiers, Deep learning (EEGNet)
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - FINAL ENHANCED VERSION")
print("="*60)

# ============================================================
# SYNTHETIC DATA - Hard but learnable
# ============================================================
print("\n[1] Generating motor imagery data...")

fs = 128
n_trials = 250
n_channels = 4

def generate_eeg_background(n_samples, fs):
    """Generate realistic EEG background"""
    t = np.arange(0, n_samples/fs, 1/fs)
    signal = np.zeros(n_samples)
    
    # Multiple frequencies
    for freq in [6, 8, 10, 12, 15, 20]:
        amp = np.random.uniform(3, 10)
        phase = np.random.uniform(0, 2*np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Pink noise
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    pink = np.random.randn(len(freqs)) / (freqs + 0.5)**0.6
    signal += np.fft.irfft(pink, n_samples) * 6
    
    return signal

def generate_motor_imagery_trial(label, fs=128, t_len=4):
    """Generate motor imagery with realistic ERD/ERS patterns"""
    n_samples = int(fs * t_len)
    X = np.zeros((n_channels, n_samples))
    
    # Background for each channel
    for ch in range(n_channels):
        X[ch] = generate_eeg_background(n_samples, fs)
    
    # Motor imagery effect (ERD)
    t = np.arange(0, t_len, 1/fs)
    
    # Mu rhythm envelope
    mu_envelope = np.sin(2 * np.pi * 10 * t)**2
    
    if label == 0:  # LEFT hand
        # Suppress alpha on C4 (right motor, ch1)
        X[1] = X[1] * (1 - 0.4 * mu_envelope)
        # Add slight beta increase on C3 (left, ch0) - ERS
        X[0] = X[0] * (1 + 0.15 * mu_envelope)
    else:  # RIGHT hand
        # Suppress alpha on C3 (left motor, ch0)
        X[0] = X[0] * (1 - 0.4 * mu_envelope)
        # Slight beta on C4
        X[1] = X[1] * (1 + 0.15 * mu_envelope)
    
    # Add noise
    for ch in range(n_channels):
        X[ch] += np.random.randn(n_samples) * np.random.uniform(3, 8)
    
    # 3% label noise
    if np.random.random() < 0.03:
        label = 1 - label
    
    return X, label

# Generate data
X, y = [], []
for i in range(n_trials):
    label = np.random.randint(0, 2)
    trial, label = generate_motor_imagery_trial(label)
    X.append(trial)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data: {X.shape} | Left: {np.sum(np.array(y)==0)}, Right: {np.sum(np.array(y)==1)}")

# ============================================================
# CSP FEATURES
# ============================================================
print("\n[2] Extracting CSP features...")

def bandpass(X, fs, lo=8, hi=30):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    return np.array([filtfilt(b, a, trial, axis=1) for trial in X])

def csp_features(X, y, n_filters=3):
    """Compute CSP"""
    X0, X1 = X[np.array(y)==0], X[np.array(y)==1]
    cov0 = np.mean([np.cov(t) for t in X0], axis=0)
    cov1 = np.mean([np.cov(t) for t in X1], axis=0)
    
    eigvals, eigvecs = eigh(cov0, cov1)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    filters = np.hstack([eigvecs[:, :n_filters], eigvecs[:, -n_filters:]])
    
    feats = []
    for trial in X:
        proj = filters.T @ trial
        for row in proj:
            feats.append(np.log(np.var(row) + 1e-10))
    return np.array(feats).reshape(-1, 2*n_filters)

X_csp = csp_features(bandpass(X, fs, 8, 30), y)
print(f"    CSP: {X_csp.shape}")

# ============================================================
# FREQUENCY FEATURES  
# ============================================================
print("\n[3] Extracting frequency features...")

def freq_features(X, fs=128):
    bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta_l': (13, 20), 'beta_h': (20, 30)}
    feats = []
    
    for trial in X:
        t_feats = []
        for ch in trial:
            f, p = welch(ch, fs=fs, nperseg=128)
            for b, (lo, hi) in bands.items():
                idx = (f >= lo) & (f <= hi)
                t_feats.append(np.mean(p[idx]))
            # Also: alpha/beta ratio (important for MI)
            a_idx = (f >= 8) & (f <= 13)
            b_idx = (f >= 13) & (f <= 30)
            t_feats.append(np.mean(p[a_idx]) / (np.mean(p[b_idx]) + 1e-10))
        feats.append(t_feats)
    return np.array(feats)

X_freq = freq_features(X)
print(f"    Freq: {X_freq.shape}")

# ============================================================
# TIME FEATURES
# ============================================================
print("\n[4] Extracting time features...")

def time_features(X):
    feats = []
    for trial in X:
        t_feats = []
        for ch in trial:
            t_feats.extend([
                np.mean(ch), np.std(ch), np.max(ch), np.min(ch),
                skew(ch), kurtosis(ch),
                np.percentile(ch, 25), np.percentile(ch, 75)
            ])
        feats.append(t_feats)
    return np.array(feats)

X_time = time_features(X)
print(f"    Time: {X_time.shape}")

# Combine all
X_all = np.hstack([X_csp, X_freq, X_time])
print(f"    Combined: {X_all.shape}")

# ============================================================
# TRAIN CLASSIFIERS
# ============================================================
print("\n[5] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

classifiers = {
    'Logistic': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'RF': RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
    'GB': GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

for name, clf in classifiers.items():
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    results[name] = acc
    print(f"    {name}: {acc:.2%}")

# XGBoost
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    xgb_clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, xgb_clf.predict(X_test_s))
    results['XGBoost'] = acc
    print(f"    XGBoost: {acc:.2%}")
except Exception as e:
    print(f"    XGBoost: skipped ({type(e).__name__})")

# ============================================================
# EEGNET (Deep Learning)
# ============================================================
print("\n[6] Trying EEGNet...")

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    
    print("    Training EEGNet...")
    
    # Reshape for CNN: (samples, channels, time, 1)
    X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1).astype('float32')
    X_tr, X_te, y_tr, y_te = train_test_split(X_cnn, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize
    m, s = X_tr.mean(), X_tr.std()
    X_tr, X_te = (X_tr-m)/(s+1e-8), (X_te-m)/(s+1e-8)
    
    model = Sequential([
        Conv2D(16, (1, 32), activation='relu', padding='same', input_shape=(4, 512, 1)),
        BatchNormalization(),
        MaxPooling2D((1, 4)),
        Dropout(0.3),
        
        Conv2D(32, (2, 16), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((1, 4)),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=60, batch_size=32, validation_split=0.15, callbacks=[es], verbose=0)
    
    acc = accuracy_score(y_te, (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten())
    results['EEGNet'] = acc
    print(f"    EEGNet: {acc:.2%}")
    
except ImportError:
    print("    TensorFlow not available")
except Exception as e:
    print(f"    EEGNet failed: {type(e).__name__}")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
best_name, best_acc = sorted_res[0]

print(f"\n    BEST: {best_name} = {best_acc:.2%}\n")
for name, acc in sorted_res:
    mark = " ⭐" if name == best_name else ""
    print(f"    {name}: {acc:.2%}{mark}")

# Cross-validation
print(f"\n    5-Fold CV ({best_name}):")
cv_model = classifiers.get(best_name, classifiers['RF'])
cv = cross_val_score(cv_model, X_all, y, cv=5)
print(f"        {cv.mean():.2%} (+/- {cv.std()*2:.2%})")

print("\n" + "="*60)
print("SUMMARY:")
print("  - Features: CSP + Frequency bands + Time domain")
print("  - Classifiers: LR, LDA, SVM, RF, GB, XGBoost, EEGNet")
print("  - Data: Hard synthetic MI with realistic ERD/ERS")
print("="*60)
