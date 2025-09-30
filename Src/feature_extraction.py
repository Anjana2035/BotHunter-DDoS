import pandas as pd
import numpy as np
import os

def extract_features():
    input_path = "../Data/Processed/processed.csv"
    output_path = "../Data/Processed/features.csv"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False) 

    X = df.drop("category", axis=1)
    X = X.apply(pd.to_numeric, errors='coerce') 

    X_numeric = X.select_dtypes(include=[np.number])
    X_numeric.dropna(axis=1, how='all', inplace=True)

    X_numeric["category"] = df["category"].values
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_numeric.to_csv(output_path, index=False)

    print(f"Features extracted (unscaled) and saved to {output_path}")

if __name__ == "__main__":
    extract_features()