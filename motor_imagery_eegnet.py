#!/usr/bin/env python3
"""
EEG Motor Imagery Classification - EEGNet Implementation
A lightweight CNN architecture designed for EEG
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - EEGNet")
print("="*60)

# Check for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"    TensorFlow version: {tf.__version__}")
    USE_TF = True
except ImportError:
    print("    TensorFlow not available, using PyTorch fallback")
    USE_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    print(f"    PyTorch version: {torch.__version__}")
    USE_TORCH = True
except ImportError:
    print("    PyTorch not available either")
    USE_TORCH = False

# Create harder synthetic motor imagery data
print("\n[1] Creating harder synthetic motor imagery data...")

fs = 128
t = np.arange(0, 4, 1/fs)

n_trials = 300
n_channels = 8

def make_eeg_signal(freqs, amplitudes, phases, t, noise_level=0.5):
    signal = np.zeros_like(t)
    for freq, amp, phase in zip(freqs, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    noise = np.random.randn(len(t)) * noise_level * np.std(signal)
    return signal + noise

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        freqs = [2, 7, 10, 20, 35]
        base_amp = [5, 8, 18, 10, 5]
        phases = [np.random.rand() * 2 * np.pi for _ in freqs]
        
        signal = make_eeg_signal(freqs, base_amp, phases, t, noise_level=0.5)
        
        # Subtle motor imagery effect
        if label == 0:  # Left
            if 4 <= ch <= 6:
                signal = signal * (0.75 + 0.15 * np.random.rand())
        else:  # Right
            if 1 <= ch <= 3:
                signal = signal * (0.75 + 0.15 * np.random.rand())
        
        signal += np.random.randn(len(t)) * 6
        signals.append(signal)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"    Data: {X.shape}")

# Prepare for neural network (needs channel last for some frameworks)
# Shape: (trials, channels, samples) or (trials, samples, channels)
X_input = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # (N, 8, 512)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_input, y, test_size=0.2, random_state=42
)

# Scale per channel
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Reshape for CNN: (samples, channels, time)
X_train_cnn = X_train.transpose(0, 2, 1)  # (N, 512, 8)
X_test_cnn = X_test.transpose(0, 2, 1)

print(f"    Train: {X_train_cnn.shape}, Test: {X_test_cnn.shape}")

results = {}

if USE_TF:
    print("\n[2] Training EEGNet (TensorFlow)...")
    
    # Simplified EEGNet-like architecture
    def build_eegnet_tf(input_shape, n_classes=2):
        inputs = layers.Input(shape=input_shape)
        
        # Temporal convolution
        x = layers.Conv1D(16, kernel_size=25, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Spatial convolution
        x = layers.Conv1D(32, kernel_size=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Depthwise spatial convolution
        x = layers.DepthwiseConv1D(kernel_size=8, padding='same', depth_multiplier=2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Pooling
        x = layers.AveragePooling1D(4)(x)
        x = layers.Dropout(0.5)(x)
        
        # Separable conv
        x = layers.Conv1D(64, kernel_size=8, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    model = build_eegnet_tf((512, 8))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    y_pred = model.predict(X_test_cnn).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    results['EEGNet (TF)'] = acc
    print(f"\n    EEGNet (TF) Accuracy: {acc:.2%}")

elif USE_TORCH:
    print("\n[2] Training EEGNet (PyTorch)...")
    
    class EEGNet(nn.Module):
        def __init__(self, channels=8, samples=512, n_classes=2):
            super().__init__()
            
            # Temporal
            self.conv1 = nn.Conv1d(channels, 16, 25, padding=12)
            self.bn1 = nn.BatchNorm1d(16)
            
            # Spatial
            self.conv2 = nn.Conv1d(16, 32, 1, padding=0)
            self.bn2 = nn.BatchNorm1d(32)
            
            # Depthwise
            self.depthwise = nn.Conv1d(32, 64, 8, padding=3, groups=8)
            self.bn3 = nn.BatchNorm1d(64)
            
            self.pool = nn.AvgPool1d(4)
            self.dropout = nn.Dropout(0.5)
            
            # Separable-like
            self.conv3 = nn.Conv1d(64, 64, 8, padding=3)
            self.bn4 = nn.BatchNorm1d(64)
            
            self.fc = nn.Linear(64, n_classes)
        
        def forward(self, x):
            # x: (batch, time, channels)
            x = x.transpose(1, 2)  # (batch, channels, time)
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            
            x = self.depthwise(x)
            x = self.bn3(x)
            x = torch.relu(x)
            
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.conv3(x)
            x = self.bn4(x)
            x = torch.relu(x)
            
            x = x.mean(dim=2)  # Global average
            x = self.dropout(x)
            x = self.fc(x)
            
            return x
    
    # Convert data to tensors
    X_train_t = torch.FloatTensor(X_train_cnn)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_cnn)
    y_test_t = torch.LongTensor(y_test)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model = EEGNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_test_t)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == y_test_t).float().mean().item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50 - Val Acc: {val_acc:.2%}")
    
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_pred = test_out.argmax(dim=1)
        acc = (test_pred == y_test_t).float().mean().item()
    
    results['EEGNet (PyTorch)'] = acc
    print(f"\n    EEGNet (PyTorch) Accuracy: {acc:.2%}")

else:
    print("    No deep learning framework available!")
    print("    Falling back to sklearn CNN-like approach")
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Simple approach: flatten and use gradient boosting
    X_flat = X_train_cnn.reshape(X_train_cnn.shape[0], -1)
    X_test_flat = X_test_cnn.reshape(X_test_cnn.shape[0], -1)
    
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_flat, y_train)
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    results['GradientBoosting (flatten)'] = acc
    print(f"    Accuracy: {acc:.2%}")

print("\n[3] Results Summary:")
print("-"*40)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"    {name}: {acc:.2%}")

# Save results
with open('/Users/lobter/.openclaw/workspace/EEG-Classification/results_eegnet.txt', 'w') as f:
    f.write(f"EEGNet Version Results\n")
    f.write(f"="*40 + "\n")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"{name}: {acc:.2%}\n")

print("\n" + "="*60)
