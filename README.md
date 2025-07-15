# Airbnb Berlin & Munich Price Modeling

Machine learning models for Airbnb price prediction using supervised learning on quarterly data from Berlin and Munich, Germany.

---

## Project Overview

This repository contains a complete pipeline for predicting Airbnb listing prices using real-world data from **Berlin, Germany** and **Munich, Bavaria, Germany** over the past 12 months (quarterly data). The dataset includes approximately **18 features**, which support both **regression** and **classification** modeling tasks.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AnastasiiaX/airbnb-berlin-munich-price-modeling.git
cd airbnb-berlin-munich-price-modeling
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Load the Airbnb dataset

Ensure your data folder contains the Airbnb listings (Berlin and Munich, quarterly data). If available, include .csv or .parquet formats.

## Model Implementation

Regression using Gradient Boosted Trees (XGBoost) to predict continuous price values

Classification using Random Forests to categorize price brackets or listings

Exploratory Data Analysis (EDA) for price distributions, correlations, and city comparisons

You may find all notebook files in the notebooks/ folder or scripts inside src/.

## Technologies Used

| Task            | libraries / tools                       |
| --------------- | --------------------------------------- |
| Data processing | `pandas`, `numpy`                       |
| Regression      | `xgboost`                               |
| Classification  | `scikit-learn`                          |
| Visualization   | `matplotlib`, `seaborn`                 |
| EDA             | `pandas-profiling`, `plotly` (optional) |
| Notebooks       | Jupyter Notebook                        |

## Repository Structure

```
airbnb-berlin-munich-price-modeling/
│
├── data/                    # Airbnb datasets (Berlin & Munich)
├── notebooks/               # Jupyter notebooks for EDA and modeling
│     ├── eda.ipynb
│     ├── regression.ipynb
│     └── classification.ipynb
├── src/                     # Python scripts for training and evaluation
│     ├── train_regression.py
│     ├── train_classification.py
│     └── evaluation.py
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

## Motivation & Limitations

We originally planned to analyze Finnish real estate using the Oikotie API, but changes on the website prevented API use. As an alternative, Airbnb data from Berlin and Munich offered a rich supply of structured features over time, ideal for modeling both continuous and categorical price tasks.
