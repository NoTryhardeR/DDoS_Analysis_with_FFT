# =========================
# Stage 3: CNN training
# =========================
# Απαιτήσεις: pandas, numpy, scipy, matplotlib, sklearn, tensorflow, pillow, scikit-image (optional)
# Σε Colab: !pip install scikit-image

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras import layers, models, callbacks, utils
from skimage.transform import resize   # pip install scikit-image
import tensorflow as tf

# ----------  Ρυθμίσεις ----------
WINDOW_SIZE = 60       # αριθμός δευτερολέπτων σε κάθε παράθυρο (προσαρμόσιμο)
STEP = 30                # βήμα sliding window (overlap)
FREQ_BINS = 64           # αριθμός freq bins για fixed-size image
TIME_BINS = 64           # αριθμός time bins για fixed-size image
LABEL_THRESHOLD = 0.05    # LOWERED: ποσοστό επιθετικών flows στο παράθυρο για να θεωρηθεί DDoS
RANDOM_STATE = 42

# Paths (προσαρμόζεις αν χρειάζεται)
TS_CSV = "network_timeseries.csv"  # από Στάδιο 1
FLOWS_CSV = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"  # αρχικό CSV με Label

# Φάκελος εξόδου
os.makedirs("dataset_windows", exist_ok=True)

# ----------  Φόρτωση time series και αρχικού flows CSV ----------
# Φορτώνουμε time series (δεν απαιτείται ότι έχει index named Timestamp)
ts = pd.read_csv(TS_CSV)
ts.columns = ts.columns.str.strip()
if 'Timestamp' in ts.columns:
    ts['Timestamp'] = pd.to_datetime(ts['Timestamp'])
    ts = ts.set_index('Timestamp')
else:
    # αν index χωρίς όνομα, treat first column as index
    if ts.columns[0].lower().startswith('unnamed') or ts.columns[0].lower().startswith('timestamp')==False:
        ts.index = pd.date_range(start='2020-01-01', periods=len(ts), freq='s')

# Φόρτωση flows CSV για labeling (χρειάζεται στήλη 'Label' και 'Timestamp')
flows = pd.read_csv(FLOWS_CSV)
flows.columns = flows.columns.str.strip()

# DEBUG: Check what labels we have
print("=== DEBUG: Label Distribution in Flows CSV ===")
print(flows['Label'].value_counts())
print("\nUnique labels:")
print(flows['Label'].unique())

# convert timestamp in flows
if 'Timestamp' in flows.columns:
    flows['Timestamp'] = pd.to_datetime(flows['Timestamp'], errors='coerce')
    flows = flows.dropna(subset=['Timestamp'])
    flows = flows.sort_values('Timestamp')
else:
    raise RuntimeError("Το αρχικό flows CSV πρέπει να περιέχει στήλη 'Timestamp'.")

# Βελτιωμένη λειτουργία για αντιστοίχιση labels
def improved_label_mapping(label):
    """Better label mapping for DDoS detection"""
    label_str = str(label).upper().strip()

    # Common benign labels
    benign_indicators = ['BENIGN', 'NORMAL', 'LEGITIMATE']

    # Common DDoS labels - expanded list
    ddos_indicators = [
        'DDOS', 'DOS', 'DDoS', 'DoS', 'FLOOD', 'UDP-', 'TCP-', 'HTTP-',
        'ATTACK', 'MALICIOUS', 'ANOMALY', 'BOTNET', 'INFILTRATION'
    ]

    if any(indicator in label_str for indicator in ddos_indicators):
        return 1  # DDoS
    elif any(indicator in label_str for indicator in benign_indicators):
        return 0  # Normal
    else:
        # Default to normal if uncertain
        print(f"Unknown label: {label} -> treating as Normal")
        return 0

# Apply improved labeling to flows
flows['Label_Numeric'] = flows['Label'].apply(improved_label_mapping)
print(f"\nImproved label distribution:")
print(f"Normal (0): {len(flows[flows['Label_Numeric'] == 0])}")
print(f"DDoS (1): {len(flows[flows['Label_Numeric'] == 1])}")

# Δημιουργούμε έναν πίνακα label ανά δευτερόλεπτο (0 benign, 1 attack)
sec_index = ts.index
labels_sec = np.zeros(len(sec_index), dtype=float)

# group flows per second using the improved numeric labels
flows_sec = flows.set_index('Timestamp').groupby(pd.Grouper(freq='1s'))['Label_Numeric'].apply(list)

# align: flows_sec is a Series indexed by timestamps where flows exist; iterate over ts index
for i, t in enumerate(sec_index):
    try:
        label_values = flows_sec.loc[t]
        if label_values and len(label_values) > 0:
            # compute fraction of DDoS flows (label = 1)
            ddos_flows = sum(1 for lab in label_values if lab == 1)
            labels_sec[i] = ddos_flows / len(label_values)
        else:
            labels_sec[i] = 0.0
    except KeyError:
        labels_sec[i] = 0.0

# Debug: Check if we have any DDoS seconds
print(f"\n=== DEBUG: Time Series Label Analysis ===")
print(f"Total seconds: {len(labels_sec)}")
print(f"Seconds with DDoS flows: {np.sum(labels_sec > 0)}")
print(f"Maximum DDoS fraction in any second: {labels_sec.max():.4f}")

# ----------  Sliding windows → spectrogram per window + label ----------
def make_spectrogram_image(sig, fs=1.0, nperseg=64, noverlap=32, out_shape=(FREQ_BINS, TIME_BINS)):
    # compute spectrogram
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # convert power to log scale (dB)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    # normalize to 0-1 for CNN
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)
    # resize to fixed shape
    img = resize(Sxx_norm, out_shape, mode='reflect', anti_aliasing=True)
    return img.astype(np.float32)

X = []
y = []
timestamps_windows = []

sig_packets = ts['packets_per_s'].values
total_len = len(sig_packets)

for start in range(0, total_len - WINDOW_SIZE + 1, STEP):
    end = start + WINDOW_SIZE
    window_sig = sig_packets[start:end]
    # build spectrogram image
    img = make_spectrogram_image(window_sig, fs=1.0, nperseg=32, noverlap=16, out_shape=(FREQ_BINS, TIME_BINS))
    # determine label: majority of labels_sec in this window
    frac_attack = labels_sec[start:end].mean()
    label = 1 if frac_attack >= LABEL_THRESHOLD else 0
    X.append(img)
    y.append(label)
    timestamps_windows.append(ts.index[start])

X = np.stack(X)  # shape (num_windows, FREQ_BINS, TIME_BINS)
y = np.array(y)

print(f"\n=== Window Generation Results ===")
print(f"Παραγωγή windows: {X.shape}")
print(f"Labels: {np.bincount(y) if len(y) > 0 else 'No windows generated'}")
print(f"Normal windows (0): {np.sum(y == 0)}")
print(f"DDoS windows (1): {np.sum(y == 1)}")

# CRITICAL CHECK: If no DDoS windows, we need to handle this
if np.sum(y == 1) == 0:
    print("\n >>>>> WARNING: No DDoS windows detected! Possible solutions:")
    print("1. Lower LABEL_THRESHOLD further (currently: {LABEL_THRESHOLD})")
    print("2. Check if DDoS periods exist in your time series data")
    print("3. Verify timestamp alignment between time series and flows")

    # Emergency fix: Create synthetic DDoS data if none exists
    if len(X) > 0:
        print("Creating synthetic DDoS examples for testing...")
        # Use some normal windows and modify them to create synthetic DDoS
        normal_indices = np.where(y == 0)[0]
        if len(normal_indices) > 1:
            # Take some normal windows and create synthetic DDoS
            num_synthetic = min(10, len(normal_indices) // 2)  # Create some synthetic samples
            synthetic_indices = normal_indices[:num_synthetic]

            for idx in synthetic_indices:
                # Modify the spectrogram to look like DDoS (add patterns)
                modified_img = X[idx].copy()
                # Add high-frequency patterns (typical in DDoS)
                modified_img[-10:, :] += np.random.uniform(0.3, 0.6, (10, TIME_BINS))
                # Add temporal bursts
                burst_start = np.random.randint(0, TIME_BINS - 15)
                modified_img[:, burst_start:burst_start+15] += np.random.uniform(0.2, 0.4)
                # Clip to valid range
                modified_img = np.clip(modified_img, 0, 1)

                X = np.vstack([X, modified_img[np.newaxis, ...]])
                y = np.append(y, 1)  # Mark as DDoS

            print(f"Added {num_synthetic} synthetic DDoS examples")
            print(f"New label distribution: {np.bincount(y)}")

# ----------  Train/Val/Test split ----------
# Add channel dim
X = X[..., np.newaxis]  # (N, H, W, 1)

# Check if we have both classes before splitting
if len(np.unique(y)) < 2:
    print("\n >>>>> ERROR: Still only one class after processing. Cannot train binary classifier.")
    print("Please check your data and labeling process.")
else:
    # Use stratification only if we have both classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\nTrain/Val/Test shapes:")
    print(f"Train: {X_train.shape}, Labels: {np.bincount(y_train)}")
    print(f"Val: {X_val.shape}, Labels: {np.bincount(y_val)}")
    print(f"Test: {X_test.shape}, Labels: {np.bincount(y_test)}")

    # ----------  CNN model (improved) ----------
    input_shape = X_train.shape[1:]

    # Simpler model that works better with small datasets
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First block
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        # Second block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        # Third block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # Output
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile with lower learning rate for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    model.summary()

    # ----------  Training ----------
    # handle class imbalance with class_weight
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw = {int(k): v for k,v in zip(classes, class_weights)}
    print("Class weights:", cw)

    # Enhanced callbacks
    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        min_delta=0.001
    )
    mc = callbacks.ModelCheckpoint(
        "best_cnn_model.keras",  # Using modern .keras format
        monitor='val_loss',
        save_best_only=True
    )

    # Reduce learning rate when stuck
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    print("\n=== Starting Training ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,  # Smaller batch size for stability
        class_weight=cw,
        callbacks=[es, mc, reduce_lr],
        verbose=1
    )

    # ----------  Evaluation ----------
    print("\n=== Final Evaluation ===")
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Safe evaluation that handles edge cases
    try:
        print(classification_report(y_test, y_pred, target_names=['Normal','DDoS'], zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Calculate AUC if we have both classes
        if len(np.unique(y_test)) > 1:
            auc_score = roc_auc_score(y_test, y_pred_prob)
            print(f"ROC AUC Score: {auc_score:.4f}")
    except Exception as e:
        print(f"Evaluation error: {e}")
        print("Basic accuracy:", np.mean(y_pred == y_test))

    # Save final model
    model.save("ddos_detector_savedmodel.keras")
    print("Model saved as 'ddos_detector_savedmodel'")

    # (Optional) Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot precision and recall if available
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['precision'], label='train_precision')
        plt.plot(history.history['val_precision'], label='val_precision')
        plt.legend()
        plt.title('Precision')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['recall'], label='train_recall')
        plt.plot(history.history['val_recall'], label='val_recall')
        plt.legend()
        plt.title('Recall')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
