# BotHunter

## Overview
BotHunter-DDoS is a project that detects DDoS-style attacks from flow-level network data. It takes into consideration, volumetric and aggregate features like packets and byte rates, incoming connections per destination, trains a binary classifier to classify traffic into two classes, BENIGN and ATTACK, and provides a simple interactive prediction mode for testing.

## Project Structure
```
│── Data/
│   ├── Raw/
│      └── Rawdata
│   └── Processed/
│      ├── features.csv
│      ├── heatmap.png
│      └── processed.csv
│── Models/
│   ├── label_encoder_binary.pkl
│   ├── label_encoder.pkl
│   ├── scaler_binary.pkl
│   ├── xgboost_binary.pkl
│   └── xgboost.pkl
│── Src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── prediction.py
│── requirements.txt 
│── README.md 
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Anjana2035/BotHunter-DDoS.git
cd BotHunter-DDoS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Quick Usage
```bash
cd Src
python prediction.py
```

## Dependencies

Key libraries:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib
- seaborn

Install them via `requirements.txt`.

---

This project is for educational use.