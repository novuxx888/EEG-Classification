#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Enhanced Version
- CSP (Common Spatial Patterns) features
- RandomForest & XGBoost classifiers
- Harder synthetic data
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - ENHANCED VERSION")
print("="*60)

# Create MUCH HARDER synthetic motor imagery data
print("\n[1] Creating MUCH HARDER synthetic motor imagery data...")

fs = 128  # Sampling frequency
t = np.arange(0, 3, 1/fs)  # Shorter trials = harder

n_trials = 500
n_channels = 20  # More channels = harder problem

X = []
y = []  # 0 = left, 1 = right, 2 = rest (3-class)

# Much more subtle effect
SUPPRESSION_FACTOR = 0.98  # Very subtle!

def make_eeg_signal(freqs, amplitudes, phases, t, noise_level=0.6):
    """Create mixed frequency EEG-like signal with high noise"""
    signal = np.zeros_like(t)
    for freq, amp, phase in zip(freqs, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    # Add heavy colored noise
    noise = np.random.randn(len(t)) * noise_level * np.std(signal)
    return signal + noise

for trial in range(n_trials):
    label = np.random.randint(0, 3)  # 0=left, 1=right, 2=rest
    
    signals = []
    for ch in range(n_channels):
        # Base: mix of delta, theta, alpha, beta, gamma (confusing)
        freqs = [2, 5, 8, 10, 15, 25, 35]
        base_amp = [5, 8, 15, 18, 10, 8, 5]
        phases = [np.random.rand() * 2 * np.pi for _ in freqs]
        
        signal = make_eeg_signal(freqs, base_amp, phases, t, noise_level=0.6)
        
        # Motor imagery effect (VERY subtle):
        # Left hand = slightly reduced alpha on right motor cortex
        # Right hand = slightly reduced alpha on left motor cortex
        # Rest = baseline alpha everywhere
        
        if label == 0:  # Left
            # Right hemisphere channels (assuming C4 approx channel 14-16)
            if 13 <= ch <= 16:
                signal = signal * (SUPPRESSION_FACTOR + 0.02 * np.random.rand())
        elif label == 1:  # Right
            # Left hemisphere channels (assuming C3 approx channel 3-6)
            if 3 <= ch <= 6:
                signal = signal * (SUPPRESSION_FACTOR + 0.02 * np.random.rand())
        # label == 2 is rest: no suppression
        
        # Add heavy noise and individual variation
        signal += np.random.randn(len(t)) * 10
        signals.append(signal)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Created: {X.shape}")
print(f"    Labels: {np.bincount(y)}")

# Feature extraction
print("\n[2] Extracting features...")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpower(x, fs, band, method='welch'):
    """Compute average power in a frequency band"""
    if method == 'welch':
        freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.mean(psd[idx])
    else:
        b, a = butter_bandpass(band[0], band[1], fs)
        filtered = filtfilt(b, a, x)
        return np.mean(filtered**2)

def extract_features_enhanced(X, fs=128):
    """Extract frequency band features"""
    features = []
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    for trial in X:
        trial_features = []
        
        for ch in trial:
            for band_name, band in bands.items():
                power = bandpower(ch, fs, band)
                trial_features.append(power)
            
            # Time domain
            trial_features.append(np.mean(ch))
            trial_features.append(np.std(ch))
            trial_features.append(np.max(ch) - np.min(ch))
        
        features.append(trial_features)
    
    return np.array(features)

def compute_csp(X, y, n_components=2):
    """
    Common Spatial Patterns
    Find spatial filters that maximize variance for one class vs other
    """
    # Filter in alpha/beta band
    fs = 128
    b, a = butter(4, [8/64, 30/64], btype='band')
    
    X_filtered = np.array([filtfilt(b, a, trial) for trial in X])
    
    # Compute covariance for each class
    classes = np.unique(y)
    covs = []
    
    for c in classes:
        X_c = X_filtered[y == c]
        # Covariance for each trial, then average
        n_trials = X_c.shape[0]
        cov = np.zeros((X_c.shape[1], X_c.shape[1]))
        for trial in X_c:
            trial = trial - trial.mean(axis=1, keepdims=True)
            cov += trial @ trial.T
        cov /= n_trials
        covs.append(cov)
    
    # Generalized eigenvalue problem
    cov1, cov2 = covs[0], covs[1]
    e_vals, e_vecs = eigh(cov1, cov1 + cov2)
    
    # Sort by eigenvalues (descending for class 1, ascending for class 2)
    # CSP: first few eigenvectors maximize variance for class 1
    # Last few maximize variance for class 2
    idx = np.argsort(e_vals)[::-1]
    e_vecs = e_vecs[:, idx]
    
    # Take top and bottom n_components
    csp_filters = np.hstack([e_vecs[:, :n_components], e_vecs[:, -n_components:]])
    
    return csp_filters

def extract_csp_features(X, csp_filters):
    """Project data through CSP filters and compute log-variance"""
    features = []
    
    for trial in X:
        # Apply CSP filters
        projected = csp_filters.T @ trial  # (n_filters, n_samples)
        
        # Log-variance as features
        trial_features = []
        for ch in projected:
            var = np.var(ch)
            trial_features.append(np.log(var + 1e-10))
        
        features.append(trial_features)
    
    return np.array(features)

# Extract band power features
X_features = extract_features_enhanced(X, fs)
print(f"    Band power features: {X_features.shape}")

# Compute CSP and extract CSP features
print("    Computing CSP filters...")
csp_filters = compute_csp(X, y, n_components=4)
X_csp = extract_csp_features(X, csp_filters)
print(f"    CSP features: {X_csp.shape}")

# Combine features
X_combined = np.hstack([X_features, X_csp])
print(f"    Combined features: {X_combined.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple classifiers
print("\n[3] Training classifiers...")

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"    {name}: {acc:.2%}")

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['XGBoost'] = acc
    print(f"    XGBoost: {acc:.2%}")
except ImportError:
    print("    XGBoost not available")

print("\n[4] Results Summary:")
print("-"*40)
best_clf = max(results, key=results.get)
best_acc = results[best_clf]
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " 🎯" if name == best_clf else ""
    print(f"    {name}: {acc:.2%}{marker}")

print(f"\n    Best: {best_clf} at {best_acc:.2%}")

# Save results
with open('/Users/lobter/.openclaw/workspace/EEG-Classification/results.txt', 'w') as f:
    f.write(f"Enhanced Version Results\n")
    f.write(f"="*40 + "\n")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"{name}: {acc:.2%}\n")
    f.write(f"\nBest: {best_clf} at {best_acc:.2%}\n")

print("\n" + "="*60)
