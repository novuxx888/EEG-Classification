#!/usr/bin/env python3
"""
EEG Motor Imagery - New Approach
1) CSP features
2) RandomForest/XGBoost
3) Harder synthetic data
4) EEGNet (fixed)
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - NEW APPROACH")
print("="*60)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    print("    PyTorch available!")
except:
    TORCH_AVAILABLE = False
    print("    PyTorch not available, skipping EEGNet")

# ============================================================================
# DATA GENERATION - HARDER
# ============================================================================
def create_trial(label, fs=128, difficulty='medium'):
    """Create a single trial with realistic EEG"""
    t = np.arange(0, 4, 1/fs)
    n_channels = 8
    
    # Match previous best settings
    if difficulty == 'easy':
        suppression = 0.84  # 16% suppression (matches v3 best)
        effect_prob = 0.65  # 65% effect
    elif difficulty == 'medium':
        suppression = 0.88  # 12% suppression
        effect_prob = 0.55  # 55% effect
    else:  # hard - ultra challenge
        suppression = 0.90  # 10% suppression (subtle)
        effect_prob = 0.45  # 45% effect
    
    signals = []
    for ch in range(n_channels):
        # Multi-frequency base EEG with more realistic phases
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 18
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 7
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 5
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 6
        delta = np.sin(2 * np.pi * 2 * t + np.random.rand()*2*np.pi) * 4
        base = alpha + beta1 + beta2 + theta + delta
        
        # More realistic noise
        white_noise = np.random.randn(len(t)) * 12
        drift = np.linspace(0, 3, len(t)) * np.random.randn() * 2
        # Occasional artifacts
        if np.random.rand() < 0.15:
            spike_idx = np.random.randint(0, len(t)-30)
            base[spike_idx:spike_idx+30] += np.random.randn(30) * 30
        # 50Hz line noise
        line_noise = np.sin(2 * np.pi * 50 * t) * np.random.uniform(1, 3)
        
        base += white_noise + drift + line_noise
        
        # Cross-trial and cross-channel variability
        trial_factor = np.random.uniform(0.4, 1.5)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        # Motor imagery effect - alpha suppression
        show_effect = np.random.rand() < effect_prob
        if show_effect:
            if label == 0:  # LEFT - right motor cortex (channels 2,3,6,7)
                if ch in [2, 3, 6, 7]:
                    base *= suppression + np.random.uniform(-0.08, 0.08)
            else:  # RIGHT - left motor cortex (channels 0,1,4,5)
                if ch in [0, 1, 4, 5]:
                    base *= suppression + np.random.uniform(-0.08, 0.08)
        
        signals.append(base)
    
    return np.array(signals)

def generate_data(n_trials, difficulty='medium'):
    """Generate dataset"""
    X, y = [], []
    for _ in range(n_trials):
        label = np.random.randint(0, 2)
        X.append(create_trial(label, difficulty=difficulty))
        y.append(label)
    return np.array(X), np.array(y)

# ============================================================================
# CSP FEATURES (proper implementation)
# ============================================================================
def compute_csp_filters(X, y, fs=128, n_components=3, band=(8, 13)):
    """Compute CSP spatial filters"""
    # Filter in the specified band
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    X_filt = np.array([filtfilt(b, a, trial) for trial in X])
    
    n_channels = X.shape[1]
    
    # Compute covariance matrices for each class
    covs = []
    for c in [0, 1]:
        class_cov = np.zeros((n_channels, n_channels))
        count = 0
        for trial in X_filt[y == c]:
            cov = np.cov(trial)
            # Regularization
            cov = cov + 1e-6 * np.eye(n_channels)
            class_cov += cov
            count += 1
        class_cov /= count
        covs.append(class_cov)
    
    # Solve generalized eigenvalue problem
    try:
        eigenvalues, eigenvectors = eigh(covs[0], covs[0] + covs[1])
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Take first and last n_components (most discriminative)
        # eigenvectors are COLUMNS, so we need to transpose
        top_filters = eigenvectors[:, :n_components].T  # (n_components, n_channels)
        bottom_filters = eigenvectors[:, -n_components:].T  # (n_components, n_channels)
        csp_filters = np.vstack([top_filters, bottom_filters])
        return csp_filters
    except:
        return None

def extract_csp_features(X, y, fs=128, bands=None):
    """Extract CSP features from multiple bands"""
    if bands is None:
        bands = [(8, 13), (13, 30), (18, 25)]  # mu, beta, low-beta
    
    all_features = []
    
    for band in bands:
        csp_filters = compute_csp_filters(X, y, fs, n_components=2, band=band)
        if csp_filters is None:
            continue
            
        # Project data and compute log-variance
        b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        X_filt = np.array([filtfilt(b, a, trial) for trial in X])
        
        # Project to CSP space: (n_trials, n_filters, n_times)
        # csp_filters: (n_filters=4, n_channels=8), X_filt: (n_trials, n_channels=8, n_times)
        # Transpose X_filt to (n_trials, n_times, n_channels) then matmul
        n_trials = X_filt.shape[0]
        X_csp = np.zeros((n_trials, csp_filters.shape[0], X_filt.shape[2]))
        for i in range(n_trials):
            # X_filt[i]: (8, 512), csp_filters: (4, 8) -> result: (4, 512)
            X_csp[i] = csp_filters @ X_filt[i]
        
        # Log variance features for each CSP filter
        features = np.log(np.var(X_csp, axis=2) + 1e-10)
        all_features.append(features)
    
    return np.hstack(all_features)

# ============================================================================
# STANDARD FEATURES
# ============================================================================
def extract_frequency_features(X, fs=128):
    """Extract frequency band features"""
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'mu': (8, 13),
        'beta': (13, 30),
        'low_beta': (13, 20),
        'high_beta': (20, 30)
    }
    
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            for band_name, (low, high) in bands.items():
                b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                filtered = filtfilt(b, a, ch)
                power = np.mean(filtered**2)
                trial_feats.append(power)
        features.append(trial_feats)
    
    return np.array(features)

def extract_time_features(X):
    """Extract time domain features"""
    features = []
    for trial in X:
        trial_feats = []
        for ch in trial:
            trial_feats.extend([
                np.mean(ch),
                np.std(ch),
                np.max(ch),
                np.min(ch),
                np.percentile(ch, 25),
                np.percentile(ch, 75),
                np.sqrt(np.mean(ch**2)),  # RMS
            ])
        features.append(trial_feats)
    return np.array(features)

def extract_hemisphere_features(X):
    """Extract hemisphere asymmetry features"""
    # Channels 0,1,4,5 = left hemisphere
    # Channels 2,3,6,7 = right hemisphere
    features = []
    for trial in X:
        left_power = np.mean([np.mean(trial[ch]**2) for ch in [0, 1, 4, 5]])
        right_power = np.mean([np.mean(trial[ch]**2) for ch in [2, 3, 6, 7]])
        asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
        
        # Per-band asymmetry
        trial_feats = [asymmetry]
        for low, high in [(8, 13), (13, 30)]:
            b, a = butter(4, [low/64, high/64], btype='band')
            left_b = np.mean([np.mean(filtfilt(b, a, trial[ch])**2) for ch in [0, 1, 4, 5]])
            right_b = np.mean([np.mean(filtfilt(b, a, trial[ch])**2) for ch in [2, 3, 6, 7]])
            trial_feats.append((left_b - right_b) / (left_b + right_b + 1e-10))
        
        features.append(trial_feats)
    return np.array(features)

def get_all_features(X, y, fs=128):
    """Combine all features"""
    print("    Extracting features...")
    csp_feats = extract_csp_features(X, y, fs)
    freq_feats = extract_frequency_features(X, fs)
    time_feats = extract_time_features(X)
    hemi_feats = extract_hemisphere_features(X)
    
    return np.hstack([csp_feats, freq_feats, time_feats, hemi_feats])

# ============================================================================
# EEGNET
# ============================================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_times=512):
        super().__init__()
        # Input: (batch, 1, 8, 512)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 16), padding=(0, 8)),
            nn.BatchNorm2d(16)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        # After: (batch, 32, 1, 128)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.fc(x)
        return x

def train_eegnet(X_train, y_train, X_test, y_test, epochs=30):
    """Train EEGNet - X shape: (n_trials, n_channels, n_times)"""
    # Normalize per channel
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Reshape for PyTorch: (batch, 1, channels, times)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    
    model = EEGNet(n_channels=n_channels, n_times=n_times)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()
    return accuracy_score(y_test, preds)

# ============================================================================
# MAIN
# ============================================================================
print("\n[1] GENERATING DATA")
print("-" * 40)

difficulties = {
    'easy': ('easy', 450),      # Match previous best setup (65% effect, 16% suppression)
    'medium': ('medium', 500),  # Medium difficulty
    'hard': ('hard', 600)        # Ultra hard
}

results = {}

for diff_name, (difficulty, n_trials) in difficulties.items():
    print(f"\n  Testing {diff_name.upper()} difficulty ({difficulty})...")
    
    # Generate data
    X, y = generate_data(n_trials, difficulty=difficulty)
    print(f"    Data: {X.shape}, Labels: {np.bincount(y)}")
    
    # Extract features
    X_features = get_all_features(X, y)
    print(f"    Features: {X_features.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("    Training classifiers...")
    diff_results = {}
    
    # RandomForest
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    diff_results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test_s))
    
    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, 
                            random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train_s, y_train)
        diff_results['XGBoost'] = accuracy_score(y_test, xgb.predict(X_test_s))
    except ImportError:
        diff_results['XGBoost'] = None
    
    # GradientBoosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    gb.fit(X_train_s, y_train)
    diff_results['GradientBoosting'] = accuracy_score(y_test, gb.predict(X_test_s))
    
    # ExtraTrees
    et = ExtraTreesClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    et.fit(X_train_s, y_train)
    diff_results['ExtraTrees'] = accuracy_score(y_test, et.predict(X_test_s))
    
    # SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_s, y_train)
    diff_results['SVM-RBF'] = accuracy_score(y_test, svm.predict(X_test_s))
    
    # EEGNet - needs raw EEG data (not extracted features)
    if TORCH_AVAILABLE:
        print("    Training EEGNet...")
        # Get raw EEG data for train/test (need to redo split with original X)
        _, X_test_raw, _, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Use first 80% for train, last 20% for test to match indices
        idx = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(idx)
        train_size = int(0.8 * len(X))
        X_train_raw = X[idx[:train_size]]
        y_train_raw = y[idx[:train_size]]
        X_test_raw = X[idx[train_size:]]
        y_test_raw = y[idx[train_size:]]
        
        diff_results['EEGNet'] = train_eegnet(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    results[diff_name] = diff_results
    
    print(f"\n    Results ({diff_name}):")
    for name, acc in sorted(diff_results.items(), key=lambda x: x[1] if x[1] else 0, reverse=True):
        if acc:
            print(f"      {name}: {acc:.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

best_easy = max(results['easy'].items(), key=lambda x: x[1] if x[1] else 0)
best_hard = max(results['hard'].items(), key=lambda x: x[1] if x[1] else 0)

print(f"""
Easy difficulty: {best_easy[0]} = {best_easy[1]:.1%}
Hard difficulty: {best_hard[0]} = {best_hard[1]:.1%}

All results:
""")
for diff, diff_results in results.items():
    print(f"  {diff.upper()}:")
    for name, acc in sorted(diff_results.items(), key=lambda x: x[1] if x[1] else 0, reverse=True):
        if acc:
            print(f"    {name}: {acc:.1%}")

# Save results
with open('/Users/lobter/.openclaw/workspace/EEG-Classification/results_new_approach.txt', 'w') as f:
    f.write("New Approach Results\n")
    f.write("="*40 + "\n")
    for diff, diff_results in results.items():
        f.write(f"\n{diff.upper()}:\n")
        for name, acc in sorted(diff_results.items(), key=lambda x: x[1] if x[1] else 0, reverse=True):
            if acc:
                f.write(f"  {name}: {acc:.2%}\n")

print("\n✓ Results saved to results_new_approach.txt")