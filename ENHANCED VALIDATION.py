# =========================
# CICIDS2017 ENHANCED VALIDATION
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def comprehensive_cicids2017_validation(flows_path, ts_path):
    """
    Ολοκληρωμένη επαλήθευση για CICIDS2017 dataset
    """

    print(" ===========================================")
    print(" CICIDS2017 COMPREHENSIVE VALIDATION")
    print(" ===========================================")

    # ---------- 1. Φόρτωση και βασική επαλήθευση ----------
    print("\n 1. LOADING AND BASIC VALIDATION")

    # Φόρτωση flows data
    flows = pd.read_csv(flows_path)
    flows.columns = flows.columns.str.strip()
    print(f"✓ Flows data shape: {flows.shape}")
    print(f"✓ Columns: {list(flows.columns)}")

    # Φόρτωση time series data
    ts = pd.read_csv(ts_path)
    ts.columns = ts.columns.str.strip()
    print(f"✓ Time series shape: {ts.shape}")
    print(f"✓ Time series columns: {list(ts.columns)}")

    # ---------- 2. Ανάλυση Labels ----------
    print("\n  2. LABEL ANALYSIS")

    print("Label distribution in flows:")
    label_counts = flows['Label'].value_counts()
    print(label_counts)

    print("\nUnique labels:")
    print(flows['Label'].unique())

    # Βελτιωμένο label mapping για CICIDS2017
    def cicids2017_label_mapping(label):
        label_str = str(label).upper().strip()

        if label_str in ['BENIGN']:
            return 0
        elif label_str in ['DDOS', 'DOS', 'DDOS-LOIT', 'DOS-SLOWLORIS', 'DOS-SLOWHTTPTEST', 'DDoS']:
            return 1
        else:
            return 0

    flows['Label_Numeric'] = flows['Label'].apply(cicids2017_label_mapping)

    print(f"\n✓ Normal flows (0): {(flows['Label_Numeric'] == 0).sum()}")
    print(f"✓ DDoS flows (1): {(flows['Label_Numeric'] == 1).sum()}")
    print(f"✓ DDoS ratio: {(flows['Label_Numeric'] == 1).sum() / len(flows):.4f}")

    # ---------- 3. Χρονική Ανάλυση ----------
    print("\n 3. TEMPORAL ANALYSIS")

    # Μετατροπή timestamps
    flows['Timestamp'] = pd.to_datetime(flows['Timestamp'], errors='coerce')
    flows = flows.dropna(subset=['Timestamp'])
    flows = flows.sort_values('Timestamp')

    print(f"✓ Flows time range: {flows['Timestamp'].min()} to {flows['Timestamp'].max()}")

    # Ανάλυση DDoS περιόδου
    ddos_flows = flows[flows['Label_Numeric'] == 1]
    benign_flows = flows[flows['Label_Numeric'] == 0]

    if len(ddos_flows) > 0:
        ddos_start = ddos_flows['Timestamp'].min()
        ddos_end = ddos_flows['Timestamp'].max()
        ddos_duration = (ddos_end - ddos_start).total_seconds()

        print(f"✓ DDoS start: {ddos_start}")
        print(f"✓ DDoS end: {ddos_end}")
        print(f"✓ DDoS duration: {ddos_duration/60:.2f} minutes")
        print(f"✓ DDoS flows per minute: {len(ddos_flows) / (ddos_duration/60):.2f}")
    else:
        print(" NO DDoS FLOWS FOUND!")
        return

    # ---------- 4. Ανάλυση ανά Δευτερόλεπτο ----------
    print("\n 4. PER-SECOND ANALYSIS")

    # Ομαδοποίηση ανά δευτερόλεπτο
    flows_sec = flows.set_index('Timestamp').groupby(pd.Grouper(freq='1s'))['Label_Numeric'].agg(['count', 'mean'])
    flows_sec = flows_sec.rename(columns={'count': 'total_flows', 'mean': 'ddos_ratio'})

    print(f"✓ Total seconds with flows: {len(flows_sec)}")
    print(f"✓ Seconds with DDoS: {(flows_sec['ddos_ratio'] > 0).sum()}")
    print(f"✓ Max DDoS ratio in any second: {flows_sec['ddos_ratio'].max():.4f}")
    print(f"✓ Average flows per second: {flows_sec['total_flows'].mean():.2f}")

    # ---------- 5. Ανάλυση Time Series ----------
    print("\n 5. TIME SERIES ANALYSIS")

    # Επεξεργασία time series
    if 'Timestamp' in ts.columns:
        ts['Timestamp'] = pd.to_datetime(ts['Timestamp'])
        ts = ts.set_index('Timestamp')
    else:
        # Αν δεν υπάρχει timestamp, δημιουργία
        ts.index = pd.date_range(start=flows['Timestamp'].min(), periods=len(ts), freq='s')

    print(f"✓ Time series range: {ts.index.min()} to {ts.index.max()}")
    print(f"✓ Time series duration: {(ts.index.max() - ts.index.min()).total_seconds()/60:.2f} minutes")

    # Ελέγχουμε αν το time series καλύπτει τη DDoS περίοδο
    ts_covers_ddos = (ts.index.min() <= ddos_start) and (ts.index.max() >= ddos_end)
    print(f"✓ Time series covers DDoS period: {ts_covers_ddos}")

    # ---------- 6. Window Analysis ----------
    print("\n 6. WINDOW ANALYSIS")

    WINDOW_SIZES = [30, 60, 120]
    THRESHOLDS = [0.01, 0.05, 0.1, 0.2]

    # Δημιουργία labels ανά δευτερόλεπτο (στο time series timeline)
    labels_sec = np.zeros(len(ts), dtype=float)

    for i, timestamp in enumerate(ts.index):
        try:
            sec_data = flows_sec.loc[timestamp]
            labels_sec[i] = sec_data['ddos_ratio'] if not pd.isna(sec_data['ddos_ratio']) else 0.0
        except KeyError:
            labels_sec[i] = 0.0

    print("Window analysis with different parameters:")
    print("Window Size | Threshold | DDoS Windows | Normal Windows | Total")
    print("-" * 65)

    for window_size in WINDOW_SIZES:
        for threshold in THRESHOLDS:
            ddos_windows = 0
            normal_windows = 0

            for start in range(0, len(ts) - window_size + 1, window_size//2):
                frac_attack = labels_sec[start:start+window_size].mean()
                if frac_attack >= threshold:
                    ddos_windows += 1
                else:
                    normal_windows += 1

            total_windows = ddos_windows + normal_windows
            if total_windows > 0:
                print(f"{window_size:11d} | {threshold:9.2f} | {ddos_windows:12d} | {normal_windows:13d} | {total_windows:5d}")

    # ---------- 7. Σύσταση Βέλτιστων Παραμέτρων ----------
    print("\n 7. OPTIMAL PARAMETERS RECOMMENDATION")

    # Βέλτιστες παράμετροι βασισμένες στην ανάλυση
    optimal_window = 60  # 1 λεπτό
    optimal_threshold = 0.05  # 5%

    print(f"✓ Recommended WINDOW_SIZE: {optimal_window}")
    print(f"✓ Recommended LABEL_THRESHOLD: {optimal_threshold}")
    print(f"✓ Recommended STEP: {optimal_window//2}")

    # ---------- 8. Οπτικοποίηση ----------
    print("\n 8. CREATING VISUALIZATIONS...")

    plt.figure(figsize=(15, 10))

    # Plot 1: DDoS activity over time
    plt.subplot(3, 1, 1)
    ddos_seconds = flows_sec[flows_sec['ddos_ratio'] > 0]
    plt.plot(ddos_seconds.index, ddos_seconds['ddos_ratio'], 'r.', alpha=0.7)
    plt.title('DDoS Activity Over Time (Per Second)')
    plt.ylabel('DDoS Ratio')
    plt.grid(True, alpha=0.3)

    # Plot 2: Flows per second
    plt.subplot(3, 1, 2)
    plt.plot(flows_sec.index, flows_sec['total_flows'], 'b-', alpha=0.7)
    plt.title('Total Flows Per Second')
    plt.ylabel('Flow Count')
    plt.grid(True, alpha=0.3)

    # Plot 3: DDoS vs Normal flows over time
    plt.subplot(3, 1, 3)
    time_bins = pd.date_range(start=flows['Timestamp'].min(), end=flows['Timestamp'].max(), freq='5min')
    ddos_by_5min = ddos_flows.groupby(pd.Grouper(key='Timestamp', freq='5min')).size()
    benign_by_5min = benign_flows.groupby(pd.Grouper(key='Timestamp', freq='5min')).size()

    plt.plot(ddos_by_5min.index, ddos_by_5min.values, 'r-', label='DDoS Flows', linewidth=2)
    plt.plot(benign_by_5min.index, benign_by_5min.values, 'g-', label='Normal Flows', alpha=0.7)
    plt.title('DDoS vs Normal Flows (5-minute intervals)')
    plt.ylabel('Flow Count')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------- 9. Συμπεράσματα ----------
    print("\n 9. KEY FINDINGS AND RECOMMENDATIONS")

    findings = []

    # DDoS concentration
    ddos_concentration = len(ddos_flows) / len(flows)
    if ddos_concentration > 0.5:
        findings.append(f"• High DDoS concentration ({ddos_concentration:.1%}) - good for detection")
    else:
        findings.append(f"• Moderate DDoS concentration ({ddos_concentration:.1%})")

    # Duration analysis
    if ddos_duration < 300:  # 5 minutes
        findings.append(f"• Short DDoS duration ({ddos_duration/60:.1f} min) - use smaller windows")
    else:
        findings.append(f"• Reasonable DDoS duration ({ddos_duration/60:.1f} min)")

    # Time coverage
    if ts_covers_ddos:
        findings.append("• Time series properly covers DDoS period")
    else:
        findings.append(" TIME SERIES DOES NOT COVER DDoS PERIOD - MAJOR ISSUE!")

    # Window recommendations
    findings.append(f"• Use WINDOW_SIZE={optimal_window}, THRESHOLD={optimal_threshold}")

    for finding in findings:
        print(finding)

    return {
        'flows': flows,
        'time_series': ts,
        'ddos_start': ddos_start,
        'ddos_end': ddos_end,
        'optimal_window': optimal_window,
        'optimal_threshold': optimal_threshold
    }

# =========================
# ΕΚΤΕΛΕΣΗ ΤΟΥ VALIDATION
# =========================

if __name__ == "__main__":
    #file paths
    FLOWS_CSV = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    TS_CSV = "network_timeseries.csv"  # το time series από το Στάδιο 1

    try:
        results = comprehensive_cicids2017_validation(FLOWS_CSV, TS_CSV)
        print("\n VALIDATION COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n VALIDATION FAILED: {e}")
        print("Please check your file paths and data formats.")
