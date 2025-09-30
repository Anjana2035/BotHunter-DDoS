import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join("..", "Models", "xgboost_binary.pkl")
SCALER_PATH = os.path.join("..", "Models", "scaler_binary.pkl")

KEPT_FEATURES = ['proto_number', 'pkts', 'bytes', 'dur', 'rate']

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) 
except FileNotFoundError as e:
    print(f"[ERROR] Could not load model component: {e}. ")
    print("Please make sure that you are in Src directory. If not, run: cd Src in terminal and try again.")
    exit()

print("\nBotHunter Interactive Predictor (Final Model)")
print("\n Enter network flow details to predict if the flow is BENIGN or ATTACK.\n")

try:
    src_ip = input("SourceIP (e.g., 192.168.1.10): ")
    dst_ip = input("DestinationIP (e.g., 10.0.0.5): ")
    proto = int(input("Protocol number (e.g., 6 for TCP, 17 for UDP): "))
    sport = int(input("Source port: "))
    dport = int(input("Destination port: "))
    pkts = int(input("Total packets: "))
    bytes_ = int(input("Total bytes: "))
    dur = float(input("Duration (sec): "))
except ValueError:
    print("[ERROR] Invalid numeric input provided. Please restart.")
    exit()

rate = bytes_ / dur if dur > 0 else 0
srate = pkts / dur if dur > 0 else 0
drate = rate - srate

row = pd.DataFrame([{
    "SourceIP": src_ip,
    "DestinationIP": dst_ip,
    "proto_number": proto,
    "sport": sport,
    "dport": dport,
    "pkts": pkts,
    "bytes": bytes_,
    "dur": dur,
    "rate": rate,
    "srate": srate,
    "drate": drate,
    "spkts": 0, "dpkts": 0, "sbytes": 0, "dbytes": 0,
    "TnBPDstIP": 0, "TnP_PDstIP": 0, "N_IN_Conn_P_DstIP": 0,
    "attack": 0
}])

features_to_drop = [
    'attack', 'SourceIP', 'DestinationIP', 'sport', 'dport', 
    'TnBPDstIP', 'TnP_PDstIP', 'N_IN_Conn_P_DstIP',
    'srate', 'drate', 'spkts', 'dpkts', 'sbytes', 'dbytes'
]
row = row.drop(columns=[col for col in features_to_drop if col in row.columns], errors='ignore')


try:
    row = row[KEPT_FEATURES]
except KeyError as e:
    print(f"[ERROR] Missing feature {e} during column reordering.")
    exit()

row_scaled = scaler.transform(row)


ATTACK_THRESHOLD = 0.35

proba_attack = model.predict_proba(row_scaled)[:, 0][0] 

pred_label_index = 0 if proba_attack >= ATTACK_THRESHOLD else 1
prediction_text = 'ATTACK' if pred_label_index == 0 else 'BENIGN'

print(f"\n Predicted Class: {prediction_text:<25}")
print(f"\n Probability of ATTACK: {proba_attack:.3f}{' '*16}")

