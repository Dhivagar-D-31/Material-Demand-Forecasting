#  Materials Demand Forecasting ML 

A full-stack Machine Learning project for forecasting product demand and optimizing inventory management in supply chains.

---

##  Overview

This project predicts product demand using historical retail data and provides actionable insights for inventory planning. It integrates:

*  Machine Learning (Random Forest)
*  Flask Backend API
*  Interactive Frontend (HTML + Chart.js)

---

##  Features

*  Demand prediction using ML model
*  REST API for real-time forecasting
*  Inventory recommendations (stock, cost, safety stock)
*  Weather & seasonal impact analysis
*  Monthly demand forecasting
*  Product recommendations and clustering

---

##  Machine Learning Model

* Model: Random Forest Regressor
* Accuracy: ~99% R² Score
* Features Used:

  * Inventory Level
  * Price & Discount
  * Weather Condition
  * Seasonality
  * Competitor Pricing
  * Historical sales data

---

## Project Structure

```
materials-demand-forecasting-ml/
│
├── app.py                      # Flask backend API
├── index.html                 # Frontend UI
├── ML_project.ipynb           # ML model development
├── retail_store_inventory.csv # Dataset
├── requirements.txt           # Dependencies
├── debug_predict.py           # API testing script
├── debug_out.txt              # Output logs
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️ Clone the repository

```bash
git clone https://github.com/Dhivagar-D-31/materials-demand-forecasting-ml.git
cd materials-demand-forecasting-ml
```

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️ Run the backend

```bash
python app.py
```

### 4️ Open the application

Open in browser:

```
http://localhost:5000
```

---

##  API Endpoints

| Endpoint            | Method | Description        |
| ------------------- | ------ | ------------------ |
| `/api/health`       | GET    | Check API status   |
| `/api/dataset-info` | GET    | Dataset details    |
| `/api/sample-data`  | GET    | Sample records     |
| `/api/predict`      | POST   | Demand prediction  |
| `/api/model-info`   | GET    | Model details      |
| `/api/analytics`    | GET    | Analytics insights |

---

##  Sample Prediction Output

* Predicted Demand
* Recommended Stock
* Estimated Cost
* Monthly Forecast
* Smart Recommendations

---

##  Screenshots (Add Later)

> You can upload UI screenshots here to improve presentation.

---

##  Use Cases

* Retail inventory management
* Supply chain optimization
* Demand planning
* Pricing strategy analysis

---

##  Author

**Dhivagar D**
GitHub: https://github.com/Dhivagar-D-31

---

##  Future Improvements

* Deploy to cloud (Render / AWS)
* Add authentication
* Improve UI/UX
* Add deep learning models

---

##  Conclusion

This project demonstrates a complete end-to-end ML system combining data science, backend development, and frontend visualization for real-world applications.

---
