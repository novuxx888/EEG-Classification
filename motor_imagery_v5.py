#!/usr/bin/env python3
"""
EEG Motor Imagery - IMPROVED v5
New approaches: Enhanced CSP, Riemannian-inspired features, Feature selection, PyTorch EEGNet
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - IMPROVED v5")
print("="*60)

# ============================================================================
# 1. CREATE SYNTHETIC DATA (Same as record breaker for fair comparison)
# ============================================================================
print("\n[1] Creating synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 450
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
        
        # Trial/channel variability
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # Motor imagery effect: 60% effect, 14% suppression
        show_effect = np.random.rand() < 0.60
        suppression = 0.86 if show_effect else 1.0
        
        if label == 0:  # LEFT - right motor cortex (channels 2,3,6,7)
            if ch in [2, 3, 6, 7]:
                base *= (0.86 + np.random.uniform(-0.1, 0.1)) * suppression
        else:  # RIGHT - left motor cortex (channels 0,1,4,5)
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
# 2. ROBUST CSP FEATURES WITH MULTIPLE METHODS
# ============================================================================
print("\n[2] Computing CSP features (multiple methods)...")

def compute_robust_csp(X, y, fs, n_components=3, band=(8, 13)):
    """Robust CSP with regularization and diagonal loading"""
    # Bandpass filter
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    n_trials = len(X)
    
    # Compute covariance matrices with regularization
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        n_class = np.sum(y == c)
        for trial in X_filt[y == c]:
            # Shrinkage covariance estimator
            cov = np.cov(trial)
            shrinkage = 0.1
            cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / n_channels * np.eye(n_channels)
            class_cov += cov
        class_cov /= n_class
        covs.append(class_cov)
    
    try:
        # Regularized CSP
        reg = 1e-5
        cov0 = covs[0] + reg * np.eye(n_channels)
        cov1 = covs[1] + reg * np.eye(n_channels)
        
        # Generalized eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(cov0, cov1 + cov0 + 1e-10*np.eye(n_channels))
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top and bottom components
        W = np.hstack([eigenvectors[:, :n_components], eigenvectors[:, -n_components:]])
        
        # Compute features
        features = []
        for trial in X_filt:
            projected = W.T @ trial
            var = np.var(projected, axis=1)
            # Log-variance features
            log_var = np.log(var[:n_components] / (var[n_components:] + 1e-10) + 1e-10)
            features.append(np.concatenate([log_var, var]))
        
        return np.array(features)  # Shape: (n_trials, n_components*4)
    except Exception as e:
        print(f"    CSP error: {e}")
        return np.zeros((len(X), n_components * 4))

# 6-band FBCSP (adding high-beta)
bands = [
    (4, 8),    # Theta
    (8, 13),   # Mu (main)
    (6, 12),   # Low mu
    (13, 20),  # Beta1
    (20, 30),  # Beta2
    (28, 35),  # High beta (new)
]

fbcsp_features = []
for band in bands:
    csp_feat = compute_robust_csp(X, y, fs, n_components=3, band=band)
    fbcsp_features.append(csp_feat)

fbcsp = np.hstack(fbcsp_features)
fbcsp = np.nan_to_num(fbcsp, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    FBCSP (6 bands): {fbcsp.shape}")

# ============================================================================
# 3. ENHANCED FEATURES
# ============================================================================
print("\n[3] Extracting enhanced features...")

def extract_enhanced_band_features(X, fs):
    """Enhanced frequency band power features with more ratios"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            freqs, psd = welch(ch, fs=fs, nperseg=64)
            
            delta = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_low = np.mean(psd[(freqs >= 13) & (freqs <= 20)])
            beta_high = np.mean(psd[(freqs >= 20) & (freqs <= 30)])
            total = delta + theta + alpha + beta_low + beta_high + 1e-10
            
            # Powers
            trial_feats.extend([delta, theta, alpha, beta_low, beta_high])
            
            # Relative powers
            trial_feats.extend([
                delta/total, theta/total, alpha/total, 
                beta_low/total, beta_high/total
            ])
            
            # Important ratios for motor imagery
            trial_feats.extend([
                alpha/(beta_low + beta_high + 1e-10),  # alpha/beta
                alpha/theta,  # alpha/theta
                (alpha + theta) / (beta_low + beta_high + 1e-10),  # (alpha+theta)/beta
                (beta_low + beta_high) / (alpha + theta + 1e-10),  # beta/(alpha+theta)
                alpha / (delta + theta + 1e-10),  # alpha/slow
                (beta_low - beta_high) / (beta_low + beta_high + 1e-10),  # beta asymmetry
            ])
            
            # Time domain
            trial_feats.extend([
                np.mean(ch), np.std(ch),
                np.max(np.abs(ch)),
                np.percentile(ch, 75) - np.percentile(ch, 25),  # IQR
                np.sqrt(np.mean(ch**2)),  # RMS
                skew(ch), kurtosis(ch),  # Distribution
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_hemisphere_features(X):
    """Motor cortex specific features"""
    features = []
    for trial in X:
        trial_feats = []
        
        # Left motor cortex: channels 0,1,4,5 (C3 area)
        left_motor = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        # Right motor cortex: channels 2,3,6,7 (C4 area)
        right_motor = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        
        # Asymmetry
        asym = (left_motor - right_motor) / (left_motor + right_motor + 1e-10)
        
        # Log powers
        trial_feats.extend([
            np.log(left_motor + 1), np.log(right_motor + 1),
            left_motor / (right_motor + 1e-10),
            asym,
            # Individual channel powers
            np.mean(trial[0]**2), np.mean(trial[1]**2),
            np.mean(trial[2]**2), np.mean(trial[3]**2),
        ])
        
        # Temporal correlation between hemispheres
        left_signal = np.mean([trial[ch] for ch in [0, 1, 4, 5]], axis=0)
        right_signal = np.mean([trial[ch] for ch in [2, 3, 6, 7]], axis=0)
        corr = np.corrcoef(left_signal, right_signal)[0, 1]
        trial_feats.append(corr if not np.isnan(corr) else 0)
        
        features.append(trial_feats)
    return np.array(features)

def extract_temporal_evolution(X):
    """Track power evolution over time"""
    features = []
    n_seg = 8  # More segments
    seg_len = X.shape[2] // n_seg
    for trial in X:
        trial_feats = []
        for ch in trial:
            for seg in range(n_seg):
                start = seg * seg_len
                end = start + seg_len
                trial_feats.append(np.mean(trial[start:end]**2))
        features.append(trial_feats)
    return np.array(features)

def extract_connectivity_enhanced(X):
    """Enhanced connectivity features"""
    features = []
    for trial in X:
        trial_feats = []
        corr = np.corrcoef(trial)
        
        # Key motor pairs
        trial_feats.append(corr[0, 2])  # Left frontal - right frontal
        trial_feats.append(corr[1, 3])  # Left central - right central
        trial_feats.append(corr[0, 1])  # Left hemisphere
        trial_feats.append(corr[2, 3])  # Right hemisphere
        trial_feats.append(corr[0, 3])  # Cross
        trial_feats.append(corr[1, 2])  # Cross
        
        # Covariance matrix features (simplified Riemannian-inspired)
        cov = np.cov(trial)
        # Eigenvalues of covariance (spread)
        eVals = np.linalg.eigvalsh(cov)
        trial_feats.extend([
            np.log(eVals.max() / (eVals.min() + 1e-10)),  # Condition number
            np.sum(eVals),  # Total variance
            np.prod(eVals + 1e-10),  # Determinant proxy
        ])
        
        features.append(trial_feats)
    return np.array(features)

def extract_wavelet_features(X):
    """Simplified wavelet-like features using bandpass"""
    features = []
    # Use multiple narrow bands
    wavelet_bands = [(8, 10), (10, 12), (12, 14), (18, 22), (22, 26)]
    for trial in X:
        trial_feats = []
        for ch in trial:
            for low, high in wavelet_bands:
                b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                filtered = filtfilt(b, a, ch)
                trial_feats.append(np.mean(filtered**2))
        features.append(trial_feats)
    return np.array(features)

# Extract all features
band_features = extract_enhanced_band_features(X, fs)
hemi_features = extract_hemisphere_features(X)
temporal_features = extract_temporal_evolution(X)
connectivity_features = extract_connectivity_enhanced(X)
wavelet_features = extract_wavelet_features(X)

# Combine
X_combined = np.hstack([
    fbcsp, 
    band_features, 
    hemi_features, 
    temporal_features, 
    connectivity_features,
    wavelet_features
])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
print(f"    Combined features: {X_combined.shape}")

# ============================================================================
# 4. TRAIN CLASSIFIERS WITH FEATURE SELECTION
# ============================================================================
print("\n[4] Training classifiers...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=min(200, X_combined.shape[1]))
X_train_fs = selector.fit_transform(X_train_s, y_train)
X_test_fs = selector.transform(X_test_s)

results = {}

# Logistic Regression with different C values
for C in [0.1, 0.5, 1.0]:
    lr = LogisticRegression(random_state=42, max_iter=3000, C=C, solver='lbfgs')
    lr.fit(X_train_s, y_train)
    results[f'LR(C={C})'] = accuracy_score(y_test, lr.predict(X_test_s))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train)
results['LDA'] = accuracy_score(y_test, lda.predict(X_test_s))

# SVM with different parameters
for C in [0.5, 1.0, 2.0]:
    svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train_s, y_train)
    results[f'SVM(C={C})'] = accuracy_score(y_test, svm.predict(X_test_s))

# RandomForest with different depths
for depth in [15, 20, 25]:
    rf = RandomForestClassifier(
        n_estimators=700, 
        max_depth=depth, 
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    results[f'RF(depth={depth})'] = accuracy_score(y_test, rf.predict(X_test_s))

# ExtraTrees with different depths
for depth in [15, 20, 25]:
    et = ExtraTreesClassifier(
        n_estimators=700, 
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42, 
        n_jobs=-1
    )
    et.fit(X_train_s, y_train)
    results[f'ET(depth={depth})'] = accuracy_score(y_test, et.predict(X_test_s))

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=300, 
    max_depth=6, 
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_s, y_train)
results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))

# XGBoost
print("    Training XGBoost...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
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
        n_estimators=600, 
        max_depth=12, 
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42, 
        verbose=-1,
        n_jobs=-1
    )
    lgb_clf.fit(X_train_s, y_train)
    results['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test_s))
except Exception as e:
    print(f"    LightGBM error: {e}")

# MLP with different architectures
for hidden in [(256, 128, 64), (512, 256), (128, 64, 32)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=600, 
                        early_stopping=True, random_state=42)
    mlp.fit(X_train_s, y_train)
    results[f'MLP{hidden}'] = accuracy_score(y_test, mlp.predict(X_test_s))

# ============================================================================
# 5. PYTHON EEGNET (PyTorch)
# ============================================================================
print("\n[5] Training PyTorch EEGNet...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class EEGNet(nn.Module):
        def __init__(self, n_channels, n_times, n_classes=1):
            super().__init__()
            # Block 1
            self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16))
            self.bn1 = nn.BatchNorm2d(16)
            self.depthwise = nn.Conv2d(16, 16, (n_channels, 1), groups=16, bias=False)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(0.25)
            
            # Block 2
            self.conv2 = nn.Conv2d(16, 32, (1, 16), padding=(0, 8))
            self.bn3 = nn.BatchNorm2d(32)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(0.25)
            
            # Block 3
            self.conv3 = nn.Conv2d(32, 64, (1, 8), padding=(0, 4))
            self.bn4 = nn.BatchNorm2d(64)
            self.drop3 = nn.Dropout(0.4)
            
            # Classifier
            self.fc = nn.Sequential(
                nn.Linear(64 * 4, 128),
                nn.ELU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool1(x)
            x = self.drop1(x)
            
            x = self.conv2(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.pool2(x)
            x = self.drop2(x)
            
            x = self.conv3(x)
            x = self.bn4(x)
            x = self.drop3(x)
            
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    X_torch = X.astype(np.float32)
    X_torch = (X_torch - X_torch.mean(axis=(0,2), keepdims=True)) / (X_torch.std(axis=(0,2), keepdims=True) + 1e-10)
    
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_torch, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_t = torch.FloatTensor(X_train_t.reshape(-1, 1, n_channels, len(t)))
    X_test_t = torch.FloatTensor(X_test_t.reshape(-1, 1, n_channels, len(t)))
    y_train_t = torch.FloatTensor(y_train_t.reshape(-1, 1))
    y_test_t = torch.FloatTensor(y_test_t.reshape(-1, 1))
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = EEGNet(n_channels, len(t)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(80):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_test_t.to(device))
            val_loss = criterion(val_output, y_test_t.to(device)).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t.to(device)).cpu().numpy()
        pred_labels = (predictions > 0.5).astype(int).flatten()
        results['EEGNet-PyTorch'] = accuracy_score(y_test_t.numpy().flatten(), pred_labels)
    
    print(f"    EEGNet (PyTorch): {results['EEGNet-PyTorch']:.1%}")
    
except Exception as e:
    print(f"    PyTorch EEGNet error: {e}")

# ============================================================================
# 6. ENSEMBLE
# ============================================================================
print("\n[6] Creating optimized ensemble...")

# Find best models
best_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"    Top 5: {[f'{n}:{a:.1%}' for n, a in best_models]}")

# Build ensemble from top performers
ensemble_estimators = []
if 'rf' in dir():
    ensemble_estimators.append(('rf', rf))
if 'et' in dir():
    ensemble_estimators.append(('et', et))
if 'svm' in dir():
    ensemble_estimators.append(('svm', svm))
if 'gb' in dir():
    ensemble_estimators.append(('gb', gb))
if 'xgb_clf' in dir():
    ensemble_estimators.append(('xgb', xgb_clf))
if 'lgb_clf' in dir():
    ensemble_estimators.append(('lgb', lgb_clf))

if len(ensemble_estimators) >= 2:
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_train_s, y_train)
    results['Ensemble'] = accuracy_score(y_test, ensemble.predict(X_test_s))

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "="*60)
print("[7] RESULTS - IMPROVED v5")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results[:15]:
    marker = " 🏆" if acc == sorted_results[0][1] else ""
    print(f"    {name}: {acc:.1%}{marker}")

best_name, best_acc = sorted_results[0]
print(f"\n    BEST: {best_name} = {best_acc:.1%}")
print(f"    Previous record: 82.5%")

# Cross-validation on top models
print("\n    Cross-validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Retrain best config for CV
rf_cv = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf_cv, scaler.fit_transform(X_combined), y, cv=cv)
print(f"      RF: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

et_cv = ExtraTreesClassifier(n_estimators=700, max_depth=20, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(et_cv, scaler.fit_transform(X_combined), y, cv=cv)
print(f"      ET: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

if 'xgb_clf' in dir():
    cv_scores = cross_val_score(xgb_clf, scaler.fit_transform(X_combined), y, cv=cv)
    print(f"      XGBoost: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
