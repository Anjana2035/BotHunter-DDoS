import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier

def train_model():
    df = pd.read_csv("Data/Processed/processed.csv")

    df["binary_label"] = df["category"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

    X = df.drop(columns=["category", "binary_label"])
    y = df["binary_label"]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

    joblib.dump(model, "Models/xgboost_binary.pkl")
    joblib.dump(label_encoder, "Models/label_encoder_binary.pkl")
    print("Binary model + encoder saved in Models/")

if __name__ == "__main__":
    train_model()