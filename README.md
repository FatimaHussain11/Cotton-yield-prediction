# Cotton Crop Yield Prediction

A machine learning project that predicts cotton lint yield (Pounds/Harvested Acre) across major U.S. cotton-growing states using historical agricultural data (state, year, and fertilizer application rates for nitrogen, phosphorous, and potash).

## Overview

The notebook walks through a full regression workflow:

1. **Data cleaning** — loads the dataset and fills missing values with the per-state column mean.
2. **Exploratory analysis** — distribution plots for fertilizer application, a bar chart of top-yielding states, and a scatter plot of yield over time.
3. **Feature engineering** — encodes `State` as a numeric category and expands the feature set with polynomial features, then standardizes them.
4. **Modeling** — trains a `GradientBoostingRegressor`, tunes hyperparameters with `GridSearchCV`, and validates with k-fold cross-validation.
5. **Evaluation** — reports MAE, MSE, RMSE, R², and adjusted R², plus a predicted-vs-actual plot on the held-out test set.

## Project structure

```
crop-yield-prediction/
├── notebooks/
│   └── crop_yield_prediction.ipynb   # main analysis and modeling notebook
├── data/
│   └── README.md                      # where to place the dataset
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Setup

```bash
git clone <your-repo-url>
cd crop-yield-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then place your dataset at `data/dataset.csv` and open the notebook:

```bash
jupyter notebook notebooks/crop_yield_prediction.ipynb
```

## Results

The tuned Gradient Boosting model is evaluated with MAE, MSE, RMSE, R², and adjusted R² on a held-out test split, and validated further with 5-fold cross-validation. See the notebook's final cells for the exact scores on your data.

## License

Released under the MIT License — see [LICENSE](LICENSE).
