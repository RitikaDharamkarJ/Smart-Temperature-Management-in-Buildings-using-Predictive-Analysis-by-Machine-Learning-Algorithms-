# 🌡️ Smart Temperature Management in Buildings Using Predictive Analysis by Machine Learning Algorithms

> IEEE Published Research | 10th Annual Conference on Computational Science & Computational Intelligence (CSCI 2023)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Models-F7931E?style=flat-square&logo=scikitlearn)
![Random Forest](https://img.shields.io/badge/Random_Forest-Best_Model-brightgreen?style=flat-square)
![Decision Tree](https://img.shields.io/badge/Decision_Tree-High_Accuracy-green?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Open_Source_Data-20BEFF?style=flat-square&logo=kaggle)
![IEEE](https://img.shields.io/badge/IEEE-CSCI_2023-00629B?style=flat-square)
![Regression](https://img.shields.io/badge/Task-Regression-9B59B6?style=flat-square)

---

## 📌 Project Overview

Energy management in buildings is one of the most pressing environmental and economic challenges of our time. Heating and cooling systems alone account for a significant portion of global energy consumption — making accurate load prediction a critical tool for sustainable building design and smart energy systems.

This research applies **machine learning regression techniques** to predict the **Heating Load (HL)** and **Cooling Load (CL)** of residential buildings based on architectural and structural features. By accurately forecasting energy loads, building engineers and facility managers can proactively optimize HVAC systems, reduce energy waste, and lower operational costs.

This work was **accepted and published in the IEEE proceedings of the 10th Annual Conference on Computational Science & Computational Intelligence (CSCI 2023)**.

**Authors:** Ritika D., Yeboah J., Nti I.K.

---

## 🎯 Research Objective

> *"To predict the heating and cooling loads of buildings using regression-based machine learning models, and identify the most effective algorithm for energy load forecasting based on open-source building performance data."*

---

## 📊 Dataset

- **Source:** Kaggle (Open-Source Building Energy Efficiency Dataset)
- **Origin:** Based on Tsanas & Xifara (2012) study using Ecotect building simulation software
- **Samples:** 768 building configurations
- **Features:** 8 input features → 2 target variables

| Feature | Description |
|---------|-------------|
| Relative Compactness | Building shape compactness ratio |
| Surface Area | Total surface area (m²) |
| Wall Area | Total wall area (m²) |
| Roof Area | Roof surface area (m²) |
| Overall Height | Building height (m) |
| Orientation | Cardinal direction (2, 3, 4, 5) |
| Glazing Area | % of floor area with glazing |
| Glazing Area Distribution | Variance of glazing distribution |
| **Heating Load** ⭐ | Target variable 1 (kWh/m²) |
| **Cooling Load** ⭐ | Target variable 2 (kWh/m²) |

---

## 🏗️ Pipeline Architecture

```
Kaggle Dataset (Building Energy Efficiency)
              ↓
    Data Loading & Exploration
    • Shape, info, describe
    • Null value checks
    • Feature distributions
              ↓
    Exploratory Data Analysis
    • Correlation heatmap
    • Heating vs Cooling Load distributions
    • Feature importance analysis
              ↓
    Data Preprocessing
    • Feature scaling
    • Train/Test split (80/20)
              ↓
  ┌──────────────────────────────────────┐
  │         Model Training               │
  │  • Linear Regression (Baseline)      │
  │  • Decision Tree Regressor           │
  │  • Random Forest Regressor ⭐ Best   │
  └──────────────────────────────────────┘
              ↓
    Model Evaluation
    MAE · MSE · R² Score
              ↓
    Publication — IEEE CSCI 2023
```

---

## 📂 Repository Structure

```
Smart-Temperature-Management/
│
├── data/
│   └── ENB2012_data.xlsx              # Kaggle building energy dataset
│
├── Smart Temperature Management in
│   Buildings using Predictive Analysis
│   by Machine Learning Algorithms.ipynb   # Main research notebook
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### Step 1 — Data Loading & Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel('ENB2012_data.xlsx')

# Basic checks
print(f"Shape: {df.shape}")
print(df.describe())
print(df.isnull().sum())
```

### Step 2 — Exploratory Data Analysis
```python
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Target variable distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Y1'].hist(ax=axes[0], bins=30, color='tomato')
axes[0].set_title('Heating Load Distribution')
df['Y2'].hist(ax=axes[1], bins=30, color='steelblue')
axes[1].set_title('Cooling Load Distribution')
plt.show()
```

### Step 3 — Data Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and targets
X = df.drop(['Y1', 'Y2'], axis=1)
y_heating = df['Y1']
y_cooling = df['Y2']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_heating, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Step 4 — Model Training & Evaluation

**Linear Regression (Baseline)**
```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
```

**Decision Tree Regressor**
```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
```

**Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

**Evaluation Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  R²   : {r2:.4f}")
```

---

## 📈 Results

### Heating Load Prediction

| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| Linear Regression | - | - | - |
| Decision Tree | - | - | - |
| **Random Forest** ⭐ | **-** | **-** | **-** |

### Cooling Load Prediction

| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| Linear Regression | - | - | - |
| Decision Tree | - | - | - |
| **Random Forest** ⭐ | **-** | **-** | **-** |

> 📝 Fill in your actual metric values from the notebook results.

---

## 💡 Key Findings

- **All three models** performed effectively in predicting both Heating and Cooling Loads, validating the suitability of ML for building energy prediction
- **Decision Tree and Random Forest** significantly outperformed Linear Regression — demonstrating that building energy loads follow non-linear patterns not captured by linear models
- **Random Forest achieved the best overall performance** based on lowest MAE and MSE, and highest R² values across both target variables
- **Relative Compactness** and **Overall Height** emerged as the most influential features for energy load prediction
- The findings support the use of **tree-based ensemble methods** as the preferred approach for building energy efficiency forecasting

---

## 🌍 Real-World Impact

This research has direct applications in:
- 🏗️ **Smart Building Design** — architects can use predictions to optimize building geometry for energy efficiency before construction
- ⚡ **HVAC Optimization** — facility managers can proactively adjust heating/cooling systems based on predicted loads
- 🌱 **Sustainability Planning** — organizations can meet energy efficiency targets and reduce carbon footprints
- 💰 **Cost Reduction** — accurate load forecasting reduces energy waste and operational costs

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.9+ |
| **ML Models** | Scikit-learn (Linear Regression, Decision Tree, Random Forest) |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **Dataset** | Kaggle — ENB2012 Building Energy Efficiency |
| **Environment** | Jupyter Notebook |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/RitikaDharamkarJ/Smart-Temperature-Management-in-Buildings-using-Predictive-Analysis-by-Machine-Learning-Algorithms-.git
cd Smart-Temperature-Management-in-Buildings-using-Predictive-Analysis-by-Machine-Learning-Algorithms-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook "Smart Temperature Management in Buildings using Predictive Analysis by Machine Learning Algorithms .ipynb"
```

---

## 🔮 Future Work

- Incorporate **real-time IoT sensor data** from smart building systems for live predictions
- Explore **deep learning models** (LSTM, Transformer) for time-series energy load forecasting
- Extend the dataset to include **climate zone variables** and seasonal weather patterns
- Build a **REST API** for integration with building management systems (BMS)
- Investigate **transfer learning** across different building types and geographies

---

## 📰 Publication

This research was **accepted and published** in the IEEE proceedings of the:

> **10th Annual Conference on Computational Science & Computational Intelligence (CSCI 2023)**
>
> *Ritika D., Yeboah J., Nti I.K. (2023). Smart Temperature Management in Buildings using Predictive Analysis by Machine Learning Algorithms.*

---

## 👩‍💻 Author

**Ritika Dharamkar**
Data Scientist & ML Engineer | IEEE Published Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ritikadharamkar-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/ritikadharamkar)
[![GitHub](https://img.shields.io/badge/GitHub-RitikaDharamkarJ-black?style=flat-square&logo=github)](https://github.com/RitikaDharamkarJ)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-ff4d6d?style=flat-square)](https://ritikadharamkarj.github.io/portfolio)

---

*Published research — IEEE CSCI 2023. See full portfolio at [datascienceportfol.io/ritikadharamkar](https://www.datascienceportfol.io/ritikadharamkar)*
