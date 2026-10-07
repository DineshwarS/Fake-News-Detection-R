# Fake News Detection Using R Software

An end-to-end Machine Learning web application for classifying news articles as **REAL** or **FAKE**, built using **R (Plumber, text2vec, glmnet)** and a modern **Vanilla HTML/CSS/JavaScript** dashboard.

---

## 1. Objective

The proliferation of misinformation across social media and digital portals creates significant challenges for public discourse. The primary objective of this project is to develop an automated Fake News Detection system using statistical Natural Language Processing (NLP) and Machine Learning techniques in **R**. The system accepts raw news headlines and story bodies, transforms them into high-dimensional TF-IDF feature vectors, and classifies authenticity in real time with high statistical confidence via a high-performance REST API and an intuitive web interface.

---

## 2. Dataset

This project utilizes the benchmark **ISOT Fake News Dataset** stored in the `Dataset/` directory:

- **`Dataset/Fake.csv`**: 23,481 articles (Labeled as `FAKE`)
- **`Dataset/True.csv`**: 21,417 articles (Labeled as `REAL`)
- **Combined Volume**: 44,898 total raw news records.

### Preprocessing & Data Cleaning
- **Feature Combination**: `title` and `text` are concatenated (`paste(title, text)`) to capture both headline sensationalism and body context.
- **Deduplication**: Duplicate records and empty/NA rows are removed, yielding **39,100 unique articles** (17,903 Fake, 21,197 Real).
- **Text Normalization**:
  - Converted to lower case.
  - Stripped URLs (`https?://...` or `www....`).
  - Removed publisher wire artifacts (e.g. `reuters`) to prevent artificial data leakage.
  - Removed numbers, punctuation, and special non-alphabetic symbols.
  - Filtered standard English stopwords (175+ words).
  - Normalized consecutive whitespaces.

---

## 3. Technologies

- **Language & Environment**: R (version 4.6.1 or >= 4.0.0)
- **Machine Learning & NLP**:
  - `text2vec` (fast C++ vocabulary creation, pruning, and TF-IDF transformation)
  - `glmnet` (extremely efficient coordinate descent algorithm for regularized Logistic Regression)
  - `Matrix` (sparse matrix operations)
- **API Framework**:
  - `plumber` (declarative REST API server for R)
  - `httpuv`, `jsonlite` (HTTP micro-server and JSON serialization)
- **Frontend**:
  - Pure HTML5, CSS3, and modern Vanilla JavaScript (ES6+)
  - Google Fonts (`Inter` & `JetBrains Mono`)
  - No external Node.js, Python, or heavy UI frameworks required.

---

## 4. Algorithm & Methodology

1. **TF-IDF Vectorization**:
   - Computes Term Frequency-Inverse Document Frequency weighting for unigrams.
   - Vocabulary is derived and pruned strictly on the training set (minimum document frequency = 5, maximum document proportion = 50%, top 10,000 terms) to completely avoid data leakage into the test fold.
2. **Classification Model**:
   - **Regularized Logistic Regression (ElasticNet / L1-L2 penalty)** via `glmnet`.
   - 5-Fold cross-validation (`cv.glmnet`) on the training set to optimize the regularization parameter $\lambda$ (`lambda.1se`).
   - Generates calibrated posterior class probabilities:
     $$\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
   - Threshold $\tau = 0.5$: Output is classified as `FAKE` if probability $\ge 0.5$, else `REAL`.

---

## 5. Architecture & Workflow

```
+-------------------------------------------------------------+
|                        DATASET                              |
|         Dataset/Fake.csv  &  Dataset/True.csv               |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 train_model.R (Back End/)                   |
|  - Data cleaning & deduplication                            |
|  - 80/20 Stratified Train/Test Split (Avoid Data Leakage)    |
|  - Vocabulary Extraction & TF-IDF (text2vec, 10k terms)     |
|  - Regularized Logistic Regression fitting (glmnet)         |
|  - Evaluation on Unseen Test Split (7,820 samples)          |
|  - Saves: model.rds, vectorizer.rds, tfidf.rds, metrics.json|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   plumber.R (Back End/)                     |
|  - Loads trained artifacts into memory                      |
|  - CORS Filter (enables browser AJAX calls)                 |
|  - Endpoints: GET /health, GET /metrics, POST /predict      |
|  - Served on http://127.0.0.1:8000 via run_api.R            |
+-------------------------------------------------------------+
                              ^
                              | HTTP REST Requests
                              v
+-------------------------------------------------------------+
|                  Front End Dashboard                        |
|  - index.html, style.css, app.js                            |
|  - Live R Backend Status Pill                               |
|  - Article input textarea with instant sample loaders       |
|  - Real-time classification verdict & confidence gauge      |
|  - Dynamic metrics cards & Confusion Matrix visualizer      |
+-------------------------------------------------------------+
```

---

## 6. Directory Structure

```text
Fake news detection using R software/
├── Back End/
│   ├── models/
│   │   ├── metrics.json        # Precomputed test evaluation metrics
│   │   ├── model.rds           # Serialized trained Logistic Regression model
│   │   ├── stopwords.rds       # Serialized stopwords vector
│   │   ├── tfidf_model.rds     # Serialized TF-IDF transformer
│   │   └── vectorizer.rds      # Serialized vocabulary vectorizer
│   ├── plumber.R               # Plumber API definition & endpoints
│   ├── run_api.R               # API startup entry point
│   └── train_model.R           # End-to-end model training pipeline
├── Dataset/
│   ├── Fake.csv                # Raw Fake news articles
│   └── True.csv                # Raw True news articles
├── Front End/
│   ├── app.js                  # Client logic, API integration & DOM updates
│   ├── index.html              # Responsive college project dashboard
│   └── style.css               # Clean modern design system
└── README.md                   # Project documentation & runbook
```

---

## 7. Installation & Requirements

### Required R Packages
Ensure R (version 4.0 or above) is installed. Install the necessary packages by running the following command in an R console or terminal:

```r
install.packages(c("plumber", "text2vec", "glmnet", "jsonlite", "Matrix"), repos = "https://cloud.r-project.org")
```

---

## 8. How to Train the Model

To execute data preprocessing, TF-IDF feature extraction, model fitting, and evaluation on the 80/20 train/test split:

```powershell
Rscript "Back End/train_model.R"
```
*(Or provide the full path to `Rscript.exe`, e.g. `& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" "Back End/train_model.R"`).*

This script:
1. Cleans and joins `Fake.csv` and `True.csv`.
2. Splits data into 31,280 training articles and 7,820 test articles.
3. Fits regularized Logistic Regression via 5-fold cross-validation.
4. Generates performance metrics and saves all artifacts to `Back End/models/`.

---

## 9. How to Start the R Backend

To start the Plumber REST API server:

```powershell
Rscript "Back End/run_api.R"
```

The API will bind to `http://127.0.0.1:8000`. You can also explore interactive Swagger API docs at `http://127.0.0.1:8000/__docs__/`.

---

## 10. How to Open the Frontend

You can open the frontend in any modern web browser:

1. **Direct File Open**:
   Double click or open `Front End/index.html` in your browser.
   
2. **Or via Local Web Server** (Recommended):
   ```powershell
   python -m http.server 3000 --directory "Front End"
   ```
   Then navigate to: **`http://127.0.0.1:3000`**.

---

## 11. API Usage & Specifications

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "Fake News Detection API",
  "engine": "R Plumber + glmnet",
  "model_loaded": true,
  "timestamp": "2026-09-18 21:44:09"
}
```

### 2. Model Evaluation Metrics
- **Endpoint**: `GET /metrics`
- **Response**:
```json
{
  "model_name": "Logistic Regression (ElasticNet via glmnet)",
  "feature_extractor": "TF-IDF (10,000 terms)",
  "accuracy": 98.45,
  "precision": 98.73,
  "recall": 97.81,
  "f1_score": 98.27,
  "confusion_matrix": {
    "true_positive": 3432,
    "false_positive": 44,
    "true_negative": 4267,
    "false_negative": 77,
    "total_tested": 7820
  }
}
```

### 3. Predict News
- **Endpoint**: `POST /predict`
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said..."
}
```
- **Response**:
```json
{
  "status": "success",
  "prediction": "REAL",
  "confidence": 99.75,
  "confidence_ratio": 0.9975,
  "fake_probability": 0.25,
  "real_probability": 99.75,
  "text_length_chars": 250,
  "cleaned_word_count": 24,
  "timestamp": "2026-09-18 21:32:30"
}
```

---

## 12. Actual Evaluation Results

Evaluated on **7,820 unseen test news articles** (20% held-out test split):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.45%** |
| **Precision** | **98.73%** |
| **Recall** | **97.81%** |
| **F1-Score** | **98.27%** |

### Confusion Matrix (Test Split: 7,820 Articles)

| | Predicted FAKE | Predicted REAL |
| :--- | :---: | :---: |
| **Actual FAKE** | **3,432** (True Positive) | **77** (False Negative) |
| **Actual REAL** | **44** (False Positive) | **4,267** (True Negative) |

---

## 13. Summary

- **Pure R Machine Learning**: No external cloud AI APIs (no OpenAI, Gemini, etc.) — all training and inference are computed locally with mathematical rigor using R's statistical packages.
- **High Accuracy & Generalization**: Regularized logistic regression achieves **98.45% accuracy** on unseen articles.
- **Production-Ready Architecture**: Clean separation between model training (`train_model.R`), API service (`plumber.R`), and modern web dashboard (`Front End/`).
