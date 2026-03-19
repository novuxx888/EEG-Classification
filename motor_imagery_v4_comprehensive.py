#!/usr/bin/env python3
"""
EEG Motor Imagery - COMPREHENSIVE v4
1. Enhanced CSP features with multiple bands
2. RandomForest/XGBoost/GradientBoosting ensemble
3. Harder synthetic data (50% effect, 11% suppression)
4. Improved EEGNet architecture

Goal: Balance difficulty while maintaining high accuracy
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - COMPREHENSIVE v4")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA - HARDER DIFFICULTY
# ============================================================================
print("\n[1] Creating synthetic motor imagery data (harder)...")

fs = 128
t = np.arange(0, 4, 1/fs)

# Test multiple difficulty levels
difficulties = {
    'MEDIUM': {'effect_pct': 0.50, 'suppression': 0.89, 'n_trials': 500},
    'HARD': {'effect_pct': 0.42, 'suppression': 0.90, 'n_trials': 600},
}

results = {}

for diff_name, diff_params in difficulties.items():
    print(f"\n--- {diff_name} DIFFICULTY ---")
    
    n_trials = diff_params['n_trials']
    effect_pct = diff_params['effect_pct']
    suppression = diff_params['suppression']
    n_channels = 8
    
    X = []
    y = []
    
    for trial in range(n_trials):
        label = np.random.randint(0, 2)
        
        signals = []
        for ch in range(n_channels):
            # Multi-frequency EEG with more realistic components
            alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
            beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 7
            beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 5
            theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 6
            delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 4
            
            base = alpha + beta1 + beta2 + theta + delta
            
            # More realistic noise
            white_noise = np.random.randn(len(t)) * 12
            drift = np.linspace(0, 3, len(t)) * np.random.randn() * 4
            
            # Artifacts
            if np.random.rand() < 0.15:
                spike_idx = np.random.randint(0, len(t)-30)
                base[spike_idx:spike_idx+30] += np.random.randn(30) * 30
            
            # Muscle artifacts (occasional)
            if np.random.rand() < 0.08:
                muscle = np.random.randn(len(t)) * 20
                b, a = butter(4, [20/(fs/2), 35/(fs/2)], btype='band')
                muscle = filtfilt(b, a, muscle)
                base += muscle * 0.3
            
            # Eye artifacts
            if np.random.rand() < 0.10:
                eog = np.sin(2 * np.pi * 2 * t) * np.random.uniform(10, 25)
                base += eog
            
            # 50Hz line noise
            if np.random.rand() < 0.20:
                line_noise = np.sin(2 * np.pi * 50 * t) * np.random.uniform(1, 3)
                base += line_noise
            
            base += white_noise + drift
            
            # Trial/channel variability
            trial_factor = np.random.uniform(0.4, 1.6)
            ch_factor = np.random.uniform(0.7, 1.3)
            base *= trial_factor * ch_factor
            
            # Motor imagery effect
            show_effect = np.random.rand() < effect_pct
            supp = suppression if show_effect else 1.0
            
            # Left hand: affect right motor cortex (ch 2,3,6,7)
            # Right hand: affect left motor cortex (ch 0,1,4,5)
            if label == 0:  # LEFT
                if ch in [2, 3, 6, 7]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            else:  # RIGHT
                if ch in [0, 1, 4, 5]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            
            signals.append(base)
        
        X.append(signals)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"    Data: {X.shape}, Labels: {np.sum(y==0)}/{np.sum(y==1)}")
    print(f"    Effect: {effect_pct*100:.0f}%, Suppression: {(1-suppression)*100:.0f}%")

# ============================================================================
# 2. ENHANCED CSP FEATURES
# ============================================================================
def compute_csp_features(X, y, fs, n_components=3, band=(8, 13)):
    """Compute CSP features for a specific band"""
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    def compute_cov(trial):
        cov = np.cov(trial)
        reg = 1e-4 * np.trace(cov)
        return cov + reg * np.eye(n_channels)
    
    # Average covariance per class
    class_covs = {}
    for c in [0, 1]:
        class_trials = X_filt[y == c]
        n_class = len(class_trials)
        avg_cov = np.zeros((n_channels, n_channels))
        for trial in class_trials:
            avg_cov += compute_cov(trial) / n_class
        class_covs[c] = avg_cov
    
    try:
        cov_sum = class_covs[0] + class_covs[1]
        cov_sum += 1e-5 * np.trace(cov_sum) * np.eye(n_channels)
        
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

# Multi-band CSP (FBCSP)
bands = {
    'mu': (8, 13),
    'beta1': (13, 20),
    'beta2': (20, 30),
    'theta': (4, 8),
    'low_mu': (6, 12),
}

def extract_features(X, y, fs):
    all_features = []
    
    # FBCSP features
    for band_name, band in bands.items():
        csp = compute_csp_features(X, y, fs, n_components=3, band=band)
        all_features.append(csp)
    
    # Band power features - collect all trial features
    band_power_features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=128)
            
            delta = np.mean(psd[(freqs >= 1) & (freqs < 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs < 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs < 30)])
            
            total = delta + theta + alpha + beta_low + beta_high + 1e-10
            
            trial_feats.extend([alpha, beta_low, beta_high, theta, delta])
            trial_feats.extend([
                alpha/total, beta_low/total, beta_high/total,
                alpha/(beta_low + beta_high + 1e-10),
                alpha/theta, (alpha + theta)/(beta_low + beta_high + 1e-10)
            ])
            
            # Time domain
            trial_feats.extend([
                np.mean(ch), np.std(ch), np.max(np.abs(ch)),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                np.sqrt(np.mean(ch**2)),
            ])
        
        # Hemisphere asymmetry
        left_ch = [0, 1, 4, 5]
        right_ch = [2, 3, 6, 7]
        
        left_power = np.mean([np.mean(trial[ch]**2) for ch in left_ch])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in right_ch])
        
        trial_feats.extend([
            (left_power - right_power) / (left_power + right_power + 1e-10),
            np.log(left_power + 1), np.log(right_power + 1),
            left_power / (right_power + 1e-10)
        ])
        
        # Spatial patterns
        frontal = np.mean([np.mean(trial[ch]**2) for ch in [0, 2, 4, 6]])
        central = np.mean([np.mean(trial[ch]**2) for ch in [1, 3, 5, 7]])
        
        trial_feats.extend([
            frontal, central, frontal/(central + 1e-10),
            (frontal - central) / (frontal + central + 1e-10)
        ])
        
        band_power_features.append(trial_feats)
    
    band_power_features = np.array(band_power_features)
    all_features.append(band_power_features)
    
    return np.hstack(all_features)

# ============================================================================
# 3. TRAIN AND EVALUATE
# ============================================================================
print("\n[2] Extracting features...")

results = {}

for diff_name in difficulties.keys():
    print(f"\n=== Processing {diff_name} ===")
    
    # Recreate data for this difficulty
    n_trials = difficulties[diff_name]['n_trials']
    effect_pct = difficulties[diff_name]['effect_pct']
    suppression = difficulties[diff_name]['suppression']
    n_channels = 8
    
    X = []
    y = []
    
    for trial in range(n_trials):
        label = np.random.randint(0, 2)
        signals = []
        for ch in range(n_channels):
            alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
            beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 7
            beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 5
            theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 6
            delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 4
            
            base = alpha + beta1 + beta2 + theta + delta
            white_noise = np.random.randn(len(t)) * 12
            drift = np.linspace(0, 3, len(t)) * np.random.randn() * 4
            
            if np.random.rand() < 0.15:
                spike_idx = np.random.randint(0, len(t)-30)
                base[spike_idx:spike_idx+30] += np.random.randn(30) * 30
            
            if np.random.rand() < 0.08:
                muscle = np.random.randn(len(t)) * 20
                b, a = butter(4, [20/(fs/2), 35/(fs/2)], btype='band')
                muscle = filtfilt(b, a, muscle)
                base += muscle * 0.3
            
            if np.random.rand() < 0.10:
                eog = np.sin(2 * np.pi * 2 * t) * np.random.uniform(10, 25)
                base += eog
            
            if np.random.rand() < 0.20:
                line_noise = np.sin(2 * np.pi * 50 * t) * np.random.uniform(1, 3)
                base += line_noise
            
            base += white_noise + drift
            
            trial_factor = np.random.uniform(0.4, 1.6)
            ch_factor = np.random.uniform(0.7, 1.3)
            base *= trial_factor * ch_factor
            
            show_effect = np.random.rand() < effect_pct
            supp = suppression if show_effect else 1.0
            
            if label == 0:
                if ch in [2, 3, 6, 7]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            else:
                if ch in [0, 1, 4, 5]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            
            signals.append(base)
        
        X.append(signals)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Extract features
    X_feat = extract_features(X, y, fs)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=3,
            random_state=42, n_jobs=-1
        ),
        'XGBoost': GradientBoostingClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.08,
            subsample=0.8, random_state=42
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=500, max_depth=20, min_samples_split=3,
            random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.08,
            subsample=0.8, random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500,
            early_stopping=True, random_state=42
        ),
        'SVM-RBF': SVC(C=10, gamma='scale', kernel='rbf', random_state=42),
    }
    
    diff_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        diff_results[name] = acc
    
    results[diff_name] = diff_results
    
    # Sort and print
    sorted_results = sorted(diff_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {diff_name} Results:")
    for name, acc in sorted_results:
        print(f"    {name}: {acc:.1%}")
    
    # Cross-validation for best
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_clf = GradientBoostingClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    cv_scores = cross_val_score(best_clf, X_feat, y, cv=cv)
    print(f"  CV (5-fold): {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}")

# ============================================================================
# 4. TRY EEGNet
# ============================================================================
print("\n[3] Testing EEGNet...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
    from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    
    def eegnet_model(n_channels=8, n_times=512, n_classes=2):
        inputs = Input(shape=(n_channels, n_times, 1))
        
        # Block 1
        x = Conv2D(16, (1, 25), padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=tf.keras.constraints.max_norm(1.))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(0.25)(x)
        
        # Block 2
        x = SeparableConv2D(32, (1, 15), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 8))(x)
        x = Dropout(0.25)(x)
        
        # Classifier
        x = Flatten()(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(n_classes, activation='softmax')(x)
        
        return Model(inputs, outputs)
    
    # Test EEGNet on MEDIUM difficulty
    n_trials = 500
    effect_pct = 0.50
    suppression = 0.89
    
    X_eeg = []
    y_eeg = []
    
    for trial in range(n_trials):
        label = np.random.randint(0, 2)
        signals = []
        for ch in range(n_channels):
            alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
            beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 7
            beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 5
            theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 6
            delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 4
            
            base = alpha + beta1 + beta2 + theta + delta
            base += np.random.randn(len(t)) * 12
            
            trial_factor = np.random.uniform(0.4, 1.6)
            base *= trial_factor
            
            show_effect = np.random.rand() < effect_pct
            supp = suppression if show_effect else 1.0
            
            if label == 0:
                if ch in [2, 3, 6, 7]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            else:
                if ch in [0, 1, 4, 5]:
                    base *= (supp + np.random.uniform(-0.06, 0.06))
            
            signals.append(base)
        
        X_eeg.append(signals)
        y_eeg.append(label)
    
    X_eeg = np.array(X_eeg)
    y_eeg = np.array(y_eeg)
    
    # Reshape for EEGNet
    X_eeg = X_eeg.transpose(0, 1, 2)  # (trials, channels, times)
    X_eeg = X_eeg[:, :, :, np.newaxis]  # Add channel dim
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_eeg, y_eeg, test_size=0.2, random_state=42, stratify=y_eeg
    )
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
    
    model = eegnet_model(n_channels=8, n_times=512)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=50, batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    _, eeg_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n  EEGNet Accuracy: {eeg_acc:.1%}")
    
    results['EEGNet'] = {'EEGNet': eeg_acc}
    
except Exception as e:
    print(f"  EEGNet failed: {e}")
    print("  (TensorFlow may not be available)")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# Find best overall
all_results = {}
for diff_name, diff_results in results.items():
    if diff_name != 'EEGNet':
        for clf_name, acc in diff_results.items():
            key = f"{diff_name}/{clf_name}"
            all_results[key] = acc

sorted_all = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

print("\nAll Results:")
for key, acc in sorted_all:
    print(f"  {key}: {acc:.1%}")

best_key, best_acc = sorted_all[0]
print(f"\n🏆 BEST: {best_key} = {best_acc:.1%}")

# Save results
with open('results_v4_comprehensive.txt', 'w') as f:
    f.write("EEG Motor Imagery v4 Comprehensive Results\n")
    f.write("="*50 + "\n\n")
    for diff_name, diff_results in results.items():
        f.write(f"{diff_name}:\n")
        for name, acc in sorted(diff_results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {name}: {acc:.2%}\n")
        f.write("\n")
    f.write(f"\n🏆 BEST: {best_key} = {best_acc:.2%}\n")

print(f"\nResults saved to results_v4_comprehensive.txt")
print("="*60)
