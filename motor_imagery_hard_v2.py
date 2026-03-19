#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - Hard Version with EEGNet
Very challenging synthetic data to better simulate real EEG
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
print("EEG MOTOR IMAGERY - HARD VERSION WITH EEGNet")
print("="*60)

# ============================================================================
# 1. CREATE VERY HARD SYNTHETIC MOTOR IMAGERY DATA
# ============================================================================
print("\n[1] Creating VERY HARD synthetic motor imagery data...")

fs = 128  # Sampling frequency
t = np.arange(0, 3, 1/fs)  # 3 seconds (shorter = harder)

n_trials = 200
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        # Complex base signal - multiple rhythms + noise
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 12
        beta = np.sin(2 * np.pi * 20 * t + np.random.rand()*2*np.pi) * 6
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 4
        base = alpha + beta + theta
        
        # Add various noises
        white_noise = np.random.randn(len(t)) * 8
        # Low-frequency drift
        drift = np.linspace(0, 1, len(t)) * np.random.randn() * 4
        # Occasional spikes (like eye blinks)
        if np.random.rand() < 0.1:
            spike_idx = np.random.randint(0, len(t)-10)
            base[spike_idx:spike_idx+10] += np.random.randn(10) * 20
        
        base += white_noise + drift
        
        # Cross-trial variability (MAJOR challenge!)
        trial_factor = np.random.uniform(0.5, 1.5)
        base *= trial_factor
        
        # Add channel-specific noise
        ch_noise = np.random.randn(len(t)) * np.random.uniform(3, 10)
        base += ch_noise
        
        # SUBTLE motor imagery effect (this is the key!)
        # Instead of strong suppression, use subtle amplitude modulation
        suppression = 0.85 if np.random.rand() < 0.7 else 1.0  # 70% of trials show effect
        
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
# 2. CSP FEATURES (proper implementation)
# ============================================================================
print("\n[2] Computing CSP features...")

def compute_csp(X, y, n_components=2):
    """Compute CSP filters"""
    # Bandpass filter for mu rhythm (8-13 Hz)
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    # Compute covariance
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        for trial in X_filt[y == c]:
            class_cov += np.cov(trial)
        class_cov /= np.sum(y == c)
        covs.append(class_cov)
    
    # Generalized eigenvalue
    eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top/bottom components
    W = np.hstack([eigenvectors[:, :n_components], 
                   eigenvectors[:, -n_components:]])
    return W

def extract_csp_features(X, W):
    """Extract CSP features"""
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    features = []
    for trial in X_filt:
        projected = W.T @ trial
        var = np.var(projected, axis=1)
        features.append(np.log(var + 1e-10))
    return np.array(features)

W = compute_csp(X, y, n_components=2)
csp_features = extract_csp_features(X, W)
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
            total = alpha + beta + theta + 1e-10
            
            trial_feats.extend([
                alpha, beta, theta,
                alpha/total, beta/total,
                np.mean(ch), np.std(ch), np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25)  # IQR
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
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_s, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_s))
results['RandomForest'] = acc_rf

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, 
                                 learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_s))
results['GradientBoosting'] = acc_gb

# ============================================================================
# 5. EEGNet (proper TensorFlow/Keras implementation)
# ============================================================================
print("\n[5] Training EEGNet (CNN)...")

eegnet_available = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    eegnet_available = True
except ImportError:
    print("    TensorFlow not available, trying PyTorch EEGNet...")

if eegnet_available:
    print("    Using TensorFlow EEGNet...")
    
    # Prepare data for EEGNet
    X_cnn = X.astype(np.float32)
    X_cnn = (X_cnn - X_cnn.mean()) / (X_cnn.std() + 1e-10)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reshape for CNN: (samples, channels, time, 1)
    X_train_c = X_train_c.reshape(-1, n_channels, len(t), 1)
    X_test_c = X_test_c.reshape(-1, n_channels, len(t), 1)
    
    # Simple EEGNet-like architecture
    model = keras.Sequential([
        layers.Conv2D(16, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (1, 16), padding='same'),
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
    model.fit(X_train_c, y_train_c, epochs=30, batch_size=16, 
              validation_split=0.2, verbose=0)
    
    acc_cnn = accuracy_score(y_test_c, (model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten())
    results['EEGNet (TensorFlow)'] = acc_cnn
else:
    # Try PyTorch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        print("    Using PyTorch EEGNet...")
        
        # Prepare data
        X_p = torch.FloatTensor(X)
        X_p = (X_p - X_p.mean()) / (X_p.std() + 1e-10)
        y_p = torch.FloatTensor(y)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=0.2, stratify=y_p, random_state=42)
        
        # Simple CNN in PyTorch
        class SimpleEEGNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16))
                self.conv2 = nn.Conv2d(16, 32, (n_channels, 1))
                self.pool = nn.AvgPool2d((1, 4))
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * (len(t)//4 - 3), 64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = x.unsqueeze(1)  # Add channel dim
                x = torch.relu(self.conv1(x))
                x = self.pool(torch.relu(self.conv2(x)))
                return self.fc(x)
        
        model = SimpleEEGNet()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCELoss()
        
        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=16)
        
        for epoch in range(30):
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            preds = (model(X_te).squeeze() > 0.5).float()
            acc_torch = accuracy_score(y_te.numpy(), preds.numpy())
        results['EEGNet (PyTorch)'] = acc_torch
    except ImportError:
        # Fallback to MLP
        from sklearn.neural_network import MLPClassifier
        X_mlp = X.reshape(len(X), -1)
        X_mlp = (X_mlp - X_mlp.mean()) / (X_mlp.std() + 1e-10)
        X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_mlp, y, test_size=0.2, stratify=y, random_state=42)
        
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
        mlp.fit(X_tr_m, y_tr_m)
        acc_mlp = accuracy_score(y_te_m, mlp.predict(X_te_m))
        results['MLP'] = acc_mlp
        print(f"    MLP: {acc_mlp:.1%}")

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

# Cross-validation on best
print("\n    Cross-validation scores:")
cv_results = {}
for name, clf in [('LR', lr), ('LDA', lda), ('SVM', svm), ('RF', rf), ('GB', gb)]:
    cv = cross_val_score(clf, scaler.fit_transform(X_combined), y, cv=5)
    cv_results[name] = cv.mean()
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
