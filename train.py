#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train baseline (Ridge) & advanced (HistGradientBoosting) on scraped laptops data.
Saves models to ./models and metrics.json
"""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

Path("models").mkdir(parents=True, exist_ok=True)

# Prefer CSV with images if present
data_path = Path("data/scraped_with_images.csv")
if not data_path.exists():
    data_path = Path("data/scraped_data.csv")
df = pd.read_csv(data_path)

def to_float(s):
    return (s.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
            .replace({"": np.nan, ".": np.nan}).astype(float))

# Basic cleaning
df["price"] = to_float(df["price"])
df = df.dropna(subset=["price"])
df = df[df["price"].between(1, 1_000_000)]
df["text"] = (df.get("title","").fillna("") + " " + df.get("description","").fillna("")).str.strip()

NUM = [c for c in ["reviews_count"] if c in df.columns]
CAT = []
TEXT = ["text"]

X = df[TEXT + CAT + NUM].copy()
y = df["price"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
tfms = [("tfidf_text", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2), "text")]
if CAT:
    tfms.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ]), CAT))
if NUM:
    tfms.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ]), NUM))
pre = ColumnTransformer(tfms, remainder="drop", sparse_threshold=0.3)

# Models
baseline = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0, random_state=42))])
advanced = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(random_state=42))])

baseline.fit(Xtr, ytr); pred_b = baseline.predict(Xte)
advanced.fit(Xtr, ytr); pred_a = advanced.predict(Xte)

def metrics(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }

res = {"baseline": metrics(yte, pred_b), "advanced": metrics(yte, pred_a)}

# Persist
joblib.dump(baseline, "models/baseline_linear.joblib")
joblib.dump(advanced, "models/advanced_hgb.joblib")
with open("models/metrics.json","w") as f:
    json.dump(res, f, indent=2)

print("Saved models → ./models")
print("Metrics:", res)
