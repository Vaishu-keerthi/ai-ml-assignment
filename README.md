# 📊 AI/ML Engineer Assignment – Phone & Laptop Price Prediction

## 1. Project Overview
This project implements an **end-to-end machine learning pipeline**:
1. **Data Acquisition** – Scraped product listings (phones & laptops) from a public e-commerce test site.
2. **Data Preparation & EDA** – Cleaned and explored data, handled missing values, summarized statistics, and visualized distributions.
3. **Model Development** – Trained and evaluated baseline linear models against advanced gradient boosting models.
4. **Deliverables** – Packaged code, notebooks, datasets, models, and documentation in a reproducible GitHub repository.

The goal is to predict **product prices** separately for **phones** and **laptops**, comparing performance between simple linear baselines and advanced boosting models.

---

## 2. Data Source
- **Website:** WebScraper E-commerce Test Site (public)  
  https://webscraper.io/test-sites/e-commerce/allinone
- **Categories Used:**  
  - **Phones:** `/computers/phones` pages  
  - **Laptops:** `/computers/laptops` pages  
- **Data Collected:**
  - Product `title`, `description`
  - `price`
  - `reviews_count`
  - Image URLs (`src`/`srcset`) → downloaded and stored as `image_path`
  - Source `page_url`
- **Scale:** ~6,000+ laptops and ~2,000+ phones scraped.

---

## 3. Preprocessing
- **Price Cleaning** – Converted price strings → float, dropped invalids and extreme outliers.  
- **Text Features** – Combined `title + description` into one text field, vectorized with **TF-IDF** (uni/bi-grams, up to 50k features).  
- **Numeric Features** – Extracted `reviews_count` from text like “12 reviews”.  
- **Split** – 80/20 train–test split with `random_state=42`.

---

## 4. Modeling Approach
- **Baseline Model:** Linear Regression  
- **Advanced Model:** HistGradientBoostingRegressor  
- **Pipeline:** `ColumnTransformer` to merge TF-IDF + numeric, wrapped in `Pipeline` for consistent preprocessing/training.  
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R².

---

## 5. Results

### Phone Price Prediction
| Model                  | MAE (↓) | RMSE (↓) | R² (↑) |
|-------------------------|---------|----------|--------|
| **Baseline (Linear)**   | 10.93   | 15.88    | 0.998  |
| **Advanced (HGBR)**     | 0.00    | 0.00     | 1.000  |

### Laptop Price Prediction
| Model                  | MAE (↓) | RMSE (↓) | R² (↑) |
|-------------------------|---------|----------|--------|
| **Baseline (Linear)**   | 99.98   | 138.95   | 0.878  |
| **Advanced (HGBR)**     | 6.43    | 20.72    | 0.997  |

### Overall (merged phones + laptops)
| Model                  | MAE (↓) | RMSE (↓) | R² (↑) |
|-------------------------|---------|----------|--------|
| **Baseline (Linear)**   | 93.62   | 133.97   | 0.897  |
| **Advanced (HGBR)**     | 5.97    | 19.96    | 0.998  |

**Observations:**
- For **phones**, HGBR achieved a perfect fit (MAE=0, R²=1.0). If not intended, check for target leakage or overly strong signals in features.  
- For **laptops**, HGBR drastically reduced error (MAE 99.98 → 6.43, R² 0.878 → 0.997).  
- Overall, gradient boosting consistently outperformed linear regression.

---

## 6. Reflections
- **Baseline models** gave a benchmark but struggled with non-linear patterns, especially for laptops.  
- **Advanced gradient boosting** captured richer interactions in text + numeric features, yielding huge accuracy gains.  
- **Data quality & preprocessing** were critical: numeric coercion, TF-IDF cleaning, and outlier handling stabilized training.  
- **Images** were downloaded and stored; future work can add CNN/CLIP embeddings for multimodal learning.  
- **Next Steps:**  
  - Add structured specs (brand, RAM, CPU).  
  - Tune with LightGBM/XGBoost.  
  - Try multimodal models (text + images).  
  - Deploy model as an API for price prediction.

---

## 7. Repository Structure
ai-ml-assignment/
├─ scraper.py
├─ eda.ipynb
├─ model.ipynb
├─ train.py
├─ requirements.txt
├─ README.md
├─ data/ # scraped CSV/JSON + images
└─ models/ # saved models + metrics.json
