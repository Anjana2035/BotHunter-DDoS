import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
import numpy as np

def train_model():
    df = pd.read_csv("../Data/Processed/features.csv")

    df["binary_label"] = df["category"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

    X = df.drop(columns=["category", "binary_label"])
    y = df["binary_label"]

    features_to_drop = [
        'attack', 'SourceIP', 'DestinationIP', 'sport', 'dport', 
        'TnBPDstIP', 'TnP_PDstIP', 'N_IN_Conn_P_DstIP',
        'srate', 'drate', 'spkts', 'dpkts', 'sbytes', 'dbytes'
    ]
    
    X = X.drop(columns=[col for col in features_to_drop if col in X.columns], errors='ignore')

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
    X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125, 
    random_state=42,
    stratify=y_temp
)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        # Regularization (Kept)
        learning_rate=0.01,  
        n_estimators=200,    
        max_depth=3,         
        subsample=0.7,       
        colsample_bytree=0.7, 
        reg_alpha=0.1,       
        reg_lambda=1.0,
        gamma=0.5,           
        tree_method="hist"
    )

    model.fit(
    X_train, 
    y_train,
    verbose=False 
)

    
    ATTACK_THRESHOLD = 0.24

    y_proba_attack = model.predict_proba(X_test)[:, 0] 

    y_pred = np.where(y_proba_attack >= ATTACK_THRESHOLD, 0, 1)
 
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix (Binary: BENIGN vs ATTACK)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    joblib.dump(model, "../Models/xgboost_binary.pkl")
    joblib.dump(label_encoder, "../Models/label_encoder_binary.pkl")
    joblib.dump(scaler, "../Models/scaler_binary.pkl")
    print("Model created.")

if __name__ == "__main__":
    train_model()