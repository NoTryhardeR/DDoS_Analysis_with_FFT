import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Διαβάζουμε τις εικόνες (spectrograms) από τον σωστό φάκελο
img_packets = load_img("spectrograms/spectrogram_packets.png", color_mode="grayscale", target_size=(64, 64))
img_bytes = load_img("spectrograms/spectrogram_bytes.png", color_mode="grayscale", target_size=(64, 64))

# Μετατρέπουμε τις εικόνες σε πίνακες
X_packets = img_to_array(img_packets)
X_bytes = img_to_array(img_bytes)

# Κανονικοποιούμε (0-1)
X_packets = X_packets / 255.0
X_bytes = X_bytes / 255.0

# Δημιουργούμε το dataset X και τα labels y
X = np.array([X_packets, X_bytes])
# IMPORTANT: The current labels are hardcoded for demonstration.
# In a real scenario, these labels (0 for Normal, 1 for DDoS) would be derived
# from the original dataset based on the time windows associated with each spectrogram.
# For this example, we assume the first spectrogram is Normal (0) and the second is DDoS (1).
y = np.array([0, 1])  # 0 = Normal, 1 = DDoS (παράδειγμα)

# Αποθήκευση του dataset για το CNN
np.savez("spectrogram_dataset.npz", X=X, y=y)

print("|<->| Το dataset αποθηκεύτηκε ως spectrogram_dataset.npz με σχήμα:", X.shape)
