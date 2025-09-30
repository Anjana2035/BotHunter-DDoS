import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def preprocess_data(input_file, output_file):
    print(f"Loading dataset from {input_file} ...")
    df = pd.read_csv(input_file)

    print(f"Initial shape: {df.shape}")

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    print(f"After cleaning: {df.shape}")

    if 'attack' in df.columns:
        df['attack'] = df['attack'].apply(lambda x: 1 if str(x).lower() in ['1', 'attack', 'ddos', 'portscan'] else 0)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.savefig("../Data/Processed/heatmap.png")
    plt.show()

if __name__ == "__main__":
    input_file = "../Data/Raw/rawdata.csv"
    output_file = "../Data/Processed/processed.csv"
    preprocess_data(input_file, output_file)
