# Credit Risk Intelligence System

> **Production-grade machine learning system for real-time credit default risk prediction.**

Built with **FastAPI** · **XGBoost** · **LightGBM** · **Docker** · **Azure**

---

## Overview

The Credit Risk Intelligence System is a full-stack ML application that predicts the probability of borrower default using ensemble learning and advanced financial behavior analysis.

This is not a notebook-only project — it's engineered as a deployable production application with a real backend, containerized infrastructure, and cloud hosting.

---

## Live Features

- Real-time credit risk scoring
- Ensemble learning with XGBoost + LightGBM
- Advanced financial feature engineering
- Recall-optimized classification with tuned thresholds
- Interactive HTMX frontend
- Production FastAPI backend with Swagger/OpenAPI docs
- Dockerized deployment pipeline
- Azure App Service deployment

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.8675** |
| Recall | **0.7202** |
| PR-AUC | 0.4054 |
| Accuracy | 0.8431 |

### Why Recall?

In credit risk, **missing a risky borrower costs far more than flagging a safe one.** This system is optimized for high recall — tuned to detect delinquency early, not just classify accurately.

### Cross-Validation Stability

| Model | Mean ROC-AUC | Std Dev |
|-------|-------------|---------|
| XGBoost | 0.8602 | ±0.0053 |
| LightGBM | 0.8640 | ±0.0052 |

Low variance across folds confirms stable generalization with minimal overfitting.

---

## System Architecture

```
User Input (UI)
      │
      ▼
FastAPI Backend
      │
      ▼
Feature Engineering Pipeline
      │
      ▼
XGBoost + LightGBM Ensemble
      │
      ▼
Risk Classification & Score
```

---

## Feature Engineering

Domain-inspired features designed to surface hidden financial stress signals:

| Feature | What It Captures |
|---------|-----------------|
| `TotalLatePayments` | Cumulative delinquency history |
| `SevereLateRatio` | Proportion of serious delinquencies |
| `HighUtilization` | Credit utilization pressure |
| `IncomeToDebt` | Repayment capacity |
| `DebtPerPerson` | Household debt burden |
| `LoanDensity` | Credit inquiry concentration |
| `RealEstateRatio` | Asset-to-debt balance |
| `HasLatePayment` | Binary delinquency flag |
| `HighDebt` | Debt overload signal |
| `LowIncome` | Income vulnerability flag |

---

## Tech Stack

**Machine Learning** — Python, Scikit-learn, XGBoost, LightGBM, Pandas, NumPy

**Backend** — FastAPI, Pydantic, Jinja2

**Frontend** — HTML5, CSS3, HTMX

**DevOps** — Docker, Docker Hub, Azure App Service

---

## Project Structure

```
credit-risk-scoring-system/
│
├── app/
│   ├── main.py            # FastAPI app entry point
│   ├── pipeline.py        # Inference pipeline
│   ├── preprocessing.py   # Feature engineering
│   ├── schemas.py         # Pydantic request/response models
│   ├── templates/
│   │   └── index.html     # HTMX frontend
│   └── static/
│       └── styles.css
│
├── artifacts/
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   └── config.json
│
├── notebooks/             # Experimentation & EDA
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

**Run locally with Docker:**

```bash
docker pull hermitsdocker/credit-risk-system
docker run -p 8000:8000 credit-risk-system
```

Then visit `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the API.

**Or run directly:**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Business Problem

Traditional credit scoring systems struggle with:

- **Imbalanced data** — defaults are rare events
- **False safety** — risky borrowers misclassified as safe
- **Deployment gaps** — models that never leave notebooks

This system addresses all three: ensemble learning for imbalance, recall optimization for risk sensitivity, and full production deployment from day one.













