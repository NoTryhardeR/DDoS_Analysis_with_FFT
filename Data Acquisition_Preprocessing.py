#Ανέβασμα του αρχείου στο Colab
from google.colab import files
uploaded = files.upload()
import pandas as pd
import matplotlib.pyplot as plt

# 2. Φόρτωση CSV
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# 3. Έλεγχος βασικών πληροφοριών & καθαρισμός
print("Διαστάσεις dataset:", df.shape)
print("\nΠρώτες γραμμές:")
print(df.head())
print("\nΟνόματα στηλών:")
print(df.columns.tolist())

# Αφαίρεση NaN & διπλοεγγραφών
df = df.dropna().drop_duplicates()
print(f"Διαστάσεις μετά τον καθαρισμό: {df.shape}")

# 4. Μετατροπή σε datetime και ταξινόμηση - ΔΙΟΡΘΩΜΕΝΟ ΟΝΟΜΑ ΣΤΗΛΗΣ
# Χρησιμοποιούμε το ακριβές όνομα της στήλης με το κενό
df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], errors='coerce')
df = df.dropna(subset=[' Timestamp'])
df = df.sort_values(' Timestamp').set_index(' Timestamp')
print(f"Τελικές διαστάσεις μετά την επεξεργασία χρονικής σήμανσης: {df.shape}")

# 5. Υπολογισμός packets/sec και bytes/sec
packets_per_s = df.resample('1s').size()  # Μέτρηση πακέτων ανά δευτερόλεπτο

# Χρησιμοποιούμε τη σωστή στήλη για bytes - 'Total Length of Fwd Packets'
bytes_per_s = df['Total Length of Fwd Packets'].resample('1s').sum()

ts_df = pd.DataFrame({
    'packets_per_s': packets_per_s,
    'bytes_per_s': bytes_per_s
}).fillna(0)

print("\nΠροεπισκόπηση χρονοσειρών:")
print(ts_df.head())
print(f"\nΔιαστάσεις χρονοσειράς: {ts_df.shape}")
print(f"Εύρος χρόνου: {ts_df.index.min()} έως {ts_df.index.max()}")

# Οπτικοποίηση χρονοσειρών
plt.figure(figsize=(12, 6))

# Δημιουργία subplots
plt.subplot(2, 1, 1)
plt.plot(ts_df.index, ts_df['packets_per_s'], color='blue', linewidth=1)
plt.title('Πακέτα ανά Δευτερόλεπτο - Friday Afternoon DDoS')
plt.ylabel('Πακέτα/δευτερόλεπτο')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(ts_df.index, ts_df['bytes_per_s'], color='red', linewidth=1)
plt.title('Bytes ανά Δευτερόλεπτο - Friday Afternoon DDoS')
plt.xlabel('Χρόνος')
plt.ylabel('Bytes/δευτερόλεπτο')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Εναλλακτική οπτικοποίηση - Κανονικοποιημένες τιμές
plt.figure(figsize=(12, 4))

# Κανονικοποίηση για καλύτερη σύγκριση
packets_normalized = (ts_df['packets_per_s'] - ts_df['packets_per_s'].min()) / (ts_df['packets_per_s'].max() - ts_df['packets_per_s'].min())
bytes_normalized = (ts_df['bytes_per_s'] - ts_df['bytes_per_s'].min()) / (ts_df['bytes_per_s'].max() - ts_df['bytes_per_s'].min())

plt.plot(ts_df.index, packets_normalized, label='Πακέτα/δευτερόλεπτο (κανονικοποιημένα)', alpha=0.7)
plt.plot(ts_df.index, bytes_normalized, label='Bytes/δευτερόλεπτο (κανονικοποιημένα)', alpha=0.7)
plt.legend()
plt.title('Κανονικοποιημένες Χρονικές Σειρές Δικτύου - Friday Afternoon DDoS')
plt.xlabel('Χρόνος')
plt.ylabel('Κανονικοποιημένη Τιμή')
plt.grid(True, alpha=0.3)
plt.show()

# Στατιστικά στοιχεία
print("\n" + "="*50)
print("ΣΤΑΤΙΣΤΙΚΑ ΣΤΟΙΧΕΙΑ ΧΡΟΝΟΣΕΙΡΑΣ")
print("="*50)
print("\nΠακέτα ανά δευτερόλεπτο:")
print(f"  Μέσος όρος: {ts_df['packets_per_s'].mean():.2f}")
print(f"  Μέγιστο: {ts_df['packets_per_s'].max()}")
print(f"  Ελάχιστο: {ts_df['packets_per_s'].min()}")
print(f"  Τυπική απόκλιση: {ts_df['packets_per_s'].std():.2f}")

print("\nBytes ανά δευτερόλεπτο:")
print(f"  Μέσος όρος: {ts_df['bytes_per_s'].mean():.2f}")
print(f"  Μέγιστο: {ts_df['bytes_per_s'].max()}")
print(f"  Ελάχιστο: {ts_df['bytes_per_s'].min()}")
print(f"  Τυπική απόκλιση: {ts_df['bytes_per_s'].std():.2f}")

# Έλεγχος για πιθανές DDoS επιθέσεις (ασυνήθιστα υψηλές τιμές)
packets_threshold = ts_df['packets_per_s'].mean() + 3 * ts_df['packets_per_s'].std()
bytes_threshold = ts_df['bytes_per_s'].mean() + 3 * ts_df['bytes_per_s'].std()

ddos_suspicious_times = ts_df[(ts_df['packets_per_s'] > packets_threshold) | (ts_df['bytes_per_s'] > bytes_threshold)]

print(f"\nΠιθανά συμβάντα DDoS (τιμές πάνω από 3*στν.απόκλιση): {len(ddos_suspicious_times)}")
if len(ddos_suspicious_times) > 0:
    print("Χρονικές στιγμές με ύποπτη δραστηριότητα:")
    print(ddos_suspicious_times)

# Αποθήκευση Αποτελεσμάτων για χρήση σε επόμενα Στάδια
ts_df.to_csv("network_timeseries.csv")
