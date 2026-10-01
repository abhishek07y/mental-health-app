# 🧠 Mental Health Score Predictor

A machine learning web app that estimates a student's **mental health score (out of 10)** from their social media usage and lifestyle habits. Built with scikit-learn and Streamlit, and deployed on Render.

**Live demo:** `<https://mental-health-app-ol45.onrender.com>`

> ⚠️ This is a learning project. The output is an ML estimate, not a medical diagnosis.

---

## Features

- Predicts a mental health score from 12 input features
- Interactive Streamlit UI with sliders and dropdowns
- Colour-coded result band (good / medium / low) with a progress bar
- Personalised suggestions based on sleep, screen time, physical activity and stress

## Dataset

**Student Social Media And Mental Health Impact** dataset (`mental_health.csv`).

| Type | Features |
|---|---|
| Numeric | `Age`, `Study_Hours`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` |
| Ordinal | `Stress_Level` (Low, Medium, High, Very High) |
| Categorical | `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `grouped_country` |
| **Target** | `Mental_Health_Score` |

## Project workflow

1. **EDA:** target distribution, correlation heatmap, stress level vs score, usage hours vs score
2. **Outlier check:** IQR method (very few outliers, so none were removed)
3. **Data cleaning:** removed duplicates and clipped negative `Physical_Activity_Hours` to 0
4. **Skewness check:** `Study_Hours` was treated as a skewed feature
5. **Feature engineering:** `Country` has 100+ unique values, so the top 10 countries were kept and the rest grouped as `Other` (`grouped_country`)
6. **Preprocessing** with `ColumnTransformer` (prevents data leakage):
   - Skewed feature: `log1p` + `StandardScaler`
   - Other numeric features: `StandardScaler`
   - Ordinal feature: `OrdinalEncoder` (Low < Medium < High < Very High)
   - Nominal features: `OneHotEncoder(handle_unknown='ignore')`
7. **Modelling** (70/30 train-test split, `random_state=42`):
   - Linear Regression (baseline)
   - Random Forest (default settings)
   - Random Forest with `RandomizedSearchCV` (15 iterations, 5-fold CV, scoring = R²)
8. **Saved** the best pipeline with `joblib`

## Results

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | `0.7237` | `0.7398` |
| Random Forest (default) | `0.9808` | `0.8776` |
| Random Forest (tuned) | `0.9685` | `0.8722` |

Replace the placeholders with the values from your notebook.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, Render

## Project structure

```
├── app.py                     # Streamlit app
├── Mental_Heath_Model.pkl     # Trained scikit-learn pipeline
├── mental_health.ipynb        # EDA, preprocessing, training
├── mental_health.csv          # Dataset
└── requirements.txt           # Dependencies
```

## Run locally

```bash
git clone https://github.com/abhishek07y/mental-health-app.git
cd mental-health-app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- **Environment variable:** `PYTHON_VERSION` set to the same version used for training

## Limitations

- Trained on a limited student dataset, so predictions may not generalise to other groups
- The score bands in the app (7+ good, 5 to 7 medium, below 5 low) are assumptions
- Not a substitute for professional mental health advice

## Author

**Abhishek** ([@abhishek07y](https://github.com/abhishek07y))
