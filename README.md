# BotHunter

## Overview
BotHunt AI is a project that detects botnets by analyzing network traffic flows. It extracts statistical and behavioral features, applies machine learning models for classification, and visualizes infected devices within communication graphs.

## Project Structure
```
│── Data/
│   ├── Raw/
│   └── Rawdata
│   └── Processed/
│── Src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── graph_analysis.py
│   ├── visualization.py
│   └── main.py
│── requirements.txt 
│── README.md 
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/BotHunt-AI.git
cd BotHunt-AI
```

### 2. Create a Virtual Environment

**On Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dependencies

Key libraries:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- networkx
- xgboost

Install them via `requirements.txt`.

---

This project is for educational use.