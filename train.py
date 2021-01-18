import lightgbm as lgb
from helpers import encode_dates, similarity_encode
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"data\iris.csv",
    parse_dates=["date"],
    index_col=[],
)
TARGET = "quantity"
df = df.dropna(subset=[TARGET])
y = df[TARGET]
X = df.drop([TARGET], axis=1)

CATEGORIZE = True
obj_cols = X.select_dtypes("object").columns
if CATEGORIZE:
    X[obj_cols] = X[obj_cols].astype("category")

X = encode_dates(X, "date")

DROPPED_FEATURES = ["id", "date_hour", "date_minute", "date_second", "lat"]
X = X.drop(DROPPED_FEATURES, axis=1)

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.0035969087735309765,
    "feature_pre_filter": False,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "num_leaves": 108,
    "feature_fraction": 0.8999999999999999,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
    "num_boost_rounds": 7323,
}
d = lgb.Dataset(X, y, silent=True)
model = lgb.train(params, d)

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    importance_type="gain",
)

