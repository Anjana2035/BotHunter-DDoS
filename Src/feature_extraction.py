import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def extract_features():
    input_path = "Data/Processed/processed.csv"
    output_path = "Data/Processed/features.csv"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    X = df.drop("category", axis=1)

    X_numeric = X.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

    X_scaled_df["category"] = df["category"].values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_scaled_df.to_csv(output_path, index=False)

    print(f"Features extracted and saved to {output_path}")

if __name__ == "__main__":
    extract_features()
