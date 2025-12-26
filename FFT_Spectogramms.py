
# --- Εισαγωγή βιβλιοθηκών ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

# ---  Φόρτωση του CSV αρχείου ---
df = pd.read_csv("network_timeseries.csv")

# Αφαίρεση κενών από ονόματα στηλών
df.columns = df.columns.str.strip()

# Μετατροπή Timestamp σε datetime και ορισμός ως index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])
df = df.set_index('Timestamp')

print(" ==> Φορτώθηκαν δεδομένα με σχήμα:\n", df.shape)
print("\nΣτήλες:", df.columns.tolist())
print(df.head())

# --- Απόσπαση των χρονοσειρών ---
signal_packets = df['packets_per_s'].values
signal_bytes = df['bytes_per_s'].values
N = len(signal_packets)
T = 1.0  # δειγματοληψία ανά δευτερόλεπτο

# --- Υπολογισμός FFT για packets/sec ---
fft_packets = fft(signal_packets)
freq = fftfreq(N, T)[:N//2]  # θετικές συχνότητες μόνο
amplitude_packets = 2.0/N * np.abs(fft_packets[0:N//2])

plt.figure(figsize=(10,4))
plt.plot(freq, amplitude_packets)
plt.title("Φάσμα Συχνοτήτων - Packets/sec")
plt.xlabel("Συχνότητα (Hz)")
plt.ylabel("Πλάτος")
plt.grid(True)
plt.show()

# ---  Δημιουργία Spectrogram για packets/sec ---
f, t, Sxx = signal.spectrogram(signal_packets, fs=1.0)

plt.figure(figsize=(10,6))
plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud')
plt.title("Spectrogram - Packets/sec")
plt.ylabel("Συχνότητα (Hz)")
plt.xlabel("Χρόνος (sec)")
plt.colorbar(label="Ισχύς (dB)")
plt.show()

# ---  Δημιουργία Spectrogram για bytes/sec ---
f2, t2, Sxx2 = signal.spectrogram(signal_bytes, fs=1.0)

plt.figure(figsize=(10,6))
plt.pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud')
plt.title("Spectrogram - Bytes/sec")
plt.ylabel("Συχνότητα (Hz)")
plt.xlabel("Χρόνος (sec)")
plt.colorbar(label="Ισχύς (dB)")
plt.show()

# ---  Αποθήκευση Spectrograms ως Εικόνες για CNN ---
os.makedirs("spectrograms", exist_ok=True)

plt.imsave("spectrograms/spectrogram_packets.png", 10*np.log10(Sxx + 1e-10))
plt.imsave("spectrograms/spectrogram_bytes.png", 10*np.log10(Sxx2 + 1e-10))


print("\nΑποθήκευση Γραφημάτων ......")
print("Τα spectrograms αποθηκεύτηκαν στον φάκελο 'spectrograms'")
