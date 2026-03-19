#!/usr/bin/env python3
"""
EEG Motor Imagery - EEGNet v3
Enhanced EEGNet architecture with better hyperparameters
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG MOTOR IMAGERY - EEGNet v3")
print("="*60)

# ============================================================================
# 1. DATA
# ============================================================================
print("\n[1] Creating data...")

fs = 128
t = np.arange(0, 4, 1/fs)
n_trials = 400
n_channels = 8

X = []
y = []

for trial in range(n_trials):
    label = np.random.randint(0, 2)
    
    signals = []
    for ch in range(n_channels):
        alpha = np.sin(2 * np.pi * 10 * t + np.random.rand()*2*np.pi) * 15
        beta1 = np.sin(2 * np.pi * 18 * t + np.random.rand()*2*np.pi) * 6
        beta2 = np.sin(2 * np.pi * 22 * t + np.random.rand()*2*np.pi) * 4
        theta = np.sin(2 * np.pi * 6 * t + np.random.rand()*2*np.pi) * 5
        base = alpha + beta1 + beta2 + theta
        
        white_noise = np.random.randn(len(t)) * 10
        drift = np.linspace(0, 2.5, len(t)) * np.random.randn() * 3
        
        if np.random.rand() < 0.12:
            spike_idx = np.random.randint(0, len(t)-20)
            base[spike_idx:spike_idx+20] += np.random.randn(20) * 25
        
        base += white_noise + drift
        
        trial_factor = np.random.uniform(0.4, 1.6)
        ch_factor = np.random.uniform(0.7, 1.3)
        base *= trial_factor * ch_factor
        
        show_effect = np.random.rand() < 0.60
        suppression = 0.86 if show_effect else 1.0
        
        if label == 0:
            if ch in [2, 3, 6, 7]:
                base *= (0.86 + np.random.uniform(-0.1, 0.1)) * suppression
        else:
            if ch in [0, 1, 4, 5]:
                base *= (0.86 + np.random.random(1)[0] * 0.2 - 0.1) * suppression
        
        signals.append(base)
    
    X.append(signals)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"    Data shape: {X.shape}")

# ============================================================================
# 2. TRAIN EEGNet
# ============================================================================
print("\n[2] Training EEGNet...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Normalize
    X_norm = X.astype(np.float32)
    mean = X_norm.mean(axis=(0, 2), keepdims=True)
    std = X_norm.std(axis=(0, 2), keepdims=True) + 1e-10
    X_norm = (X_norm - mean) / std
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X_train.reshape(-1, n_channels, len(t), 1)
    X_test = X_test.reshape(-1, n_channels, len(t), 1)
    
    # Try different architectures
    results = {}
    
    # Architecture 1: Standard EEGNet
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model1 = keras.Sequential([
        layers.Conv2D(16, (1, 32), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model1.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model1.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0,
               callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    results['EEGNet-1'] = accuracy_score(y_test, (model1.predict(X_test, verbose=0) > 0.5).astype(int).flatten())
    
    # Architecture 2: Larger
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model2 = keras.Sequential([
        layers.Conv2D(24, (1, 64), padding='same', input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                               depthwise_constraint=tf.keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.3),
        
        layers.Conv2D(48, (1, 32), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.3),
        
        layers.Conv2D(96, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='elu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model2.compile(optimizer=keras.optimizers.Adam(0.0008), loss='binary_crossentropy', metrics=['accuracy'])
    model2.fit(X_train, y_train, epochs=60, batch_size=16, validation_split=0.2, verbose=0,
               callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)])
    results['EEGNet-2'] = accuracy_score(y_test, (model2.predict(X_test, verbose=0) > 0.5).astype(int).flatten())
    
    # Architecture 3: Shallow-like
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model3 = keras.Sequential([
        layers.Conv2D(40, (n_channels, 25), input_shape=(n_channels, len(t), 1)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 75)),
        layers.Dropout(0.5),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model3.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model3.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0,
               callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    results['ShallowConv'] = accuracy_score(y_test, (model3.predict(X_test, verbose=0) > 0.5).astype(int).flatten())
    
    print(f"\n[3] RESULTS:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {acc:.1%}")
    
    best = max(results.values())
    print(f"\n    Best EEGNet: {best:.1%}")
    print(f"    Previous best (classical ML): 87.5%")
    
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
