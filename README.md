# 💰 BudgetLife — Data-Driven Analytics Dashboard

**Smart Expense Optimization Platform — Consumer Analytics & Prediction Engine**

## 🚀 Live Demo
Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting this repository.

## 📋 Overview
BudgetLife is an AI-powered financial management platform. This dashboard analyzes synthetic survey data from 2,000 Indian respondents to drive data-backed business decisions using four layers of analytics:

1. **Descriptive Analysis** — Market landscape & demographic profiling
2. **Diagnostic Analysis** — Customer segmentation (K-Means Clustering) & behavioral pattern discovery (Association Rule Mining)
3. **Predictive Analysis** — Adoption classification (Random Forest + Logistic Regression) & regression models (Ridge)
4. **Prescriptive Analysis** — Strategic recommendations, pricing strategy, & segment prioritization

## 🧠 ML Algorithms Applied
| Algorithm | Purpose |
|-----------|---------|
| **K-Means Clustering** | Segment customers by spending, saving & behavioral profiles |
| **Apriori (Association Rules)** | Discover spending pattern relationships & feature co-occurrence |
| **Random Forest + Logistic Regression** | Predict customer adoption likelihood |
| **Ridge Regression** | Predict spending, savings potential & willingness to pay |

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application (all 7 tabs, theme embedded) |
| `BudgetLife_Synthetic_Dataset.csv` | Synthetic survey dataset (2,000 respondents) |
| `requirements.txt` | Python dependencies |

## 🛠 Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dashboard Tabs
1. **📊 Overview & Descriptive** — Demographics, income/spending distributions, financial health heatmap
2. **🔬 Clustering Analysis** — Elbow method, silhouette scores, radar charts, cluster profiling
3. **🔗 Association Rule Mining** — Apriori rules with confidence, lift, support scatter plots
4. **🎯 Classification** — Model comparison, confusion matrices, ROC curves, feature importance
5. **📈 Regression Analysis** — Expenditure, savings & WTP prediction with residual analysis
6. **💡 Prescriptive Strategy** — Segment prioritization, pricing strategy, feature demand analysis
7. **🔮 New Customer Predictor** — Single entry form + bulk CSV upload for real-time predictions

## 👤 Author
Built for BudgetLife — Smart Expense Optimization Platform
