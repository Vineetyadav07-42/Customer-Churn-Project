# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project covers data preprocessing, model comparison, hyperparameter tuning, FastAPI API development, Docker containerization, and cloud deployment using Render.

## Live Demo

**Live API:** https://customer-churn-project-a4xu.onrender.com

**Swagger Documentation:** https://customer-churn-project-a4xu.onrender.com/docs

---

## Project Overview

The objective of this project is to predict customer churn based on demographic, service, contract, and billing information.

The final preprocessing and machine learning model are combined into a single scikit-learn pipeline and saved as `model.pkl`. The trained model is then exposed through a FastAPI REST API, containerized using Docker, and deployed on Render.

### Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
FastAPI
   ↓
Docker
   ↓
Render
   ↓
Live API
```

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The target variable is `Churn`:

```text
No  → 0
Yes → 1
```

The `customerID` column is removed because it does not provide useful predictive information.

### Splitting the Dataset

The dataset was split into train and test sets using scikit-learn's `train_test_split` method, with `random_state=42` and `test_size=0.2`.

### Preprocessing

Numerical features:

```text
tenure
MonthlyCharges
TotalCharges
```

are standardized using `StandardScaler`.

Categorical features are encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing steps and model are combined using a scikit-learn pipeline.

---

## Model Comparison

Four classification models were evaluated using cross-validation.

Recall and F1-score were used as the primary comparison metrics because correctly identifying customers who are likely to churn is important.

| Model                   | Mean Recall |    Mean F1 |
| ----------------------- | ----------: | ---------: |
| **Logistic Regression** |  **0.5485** | **0.5982** |
| Decision Tree           |      0.4876 |     0.4778 |
| Random Forest           |      0.4783 |     0.5417 |
| XGBoost                 |      0.5171 |     0.5560 |

### Final Model

**Logistic Regression** achieved the highest mean Recall and mean F1-score among the evaluated models.

### Fine-Tuning the Model

GridSearchCV was used to fine-tune the model. The grid used for GridSearchCV was:

```python
param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__solver": ["lbfgs", "liblinear"],
    "logisticregression__class_weight": [None, "balanced"]
}
```

The final configuration was:

```python
LogisticRegression(
    C=1,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000
)
```

`class_weight="balanced"` was used to give greater importance to the minority churn class.

---

## Final Model Evaluation

The final model was evaluated using:

* Recall
* F1-score

### Final Test Score

|   Metric   |    Score     |
| ---------: | -----------: |
|  Recall    |    0.7834    |
|  F1-Score  |    0.6136    |

> **Note:** The Recall/F1 scores above are from the held-out 20% test split, computed before the final retrain. The deployed model (`model.pkl`) was then retrained on the full dataset using the selected hyperparameters.

---

## Project Structure

```text
Customer_Churn_Prediction_Resume/
│
├── Data/
│   └── customerchurn.csv
│
├── notebooks/
│   ├── model_selection.ipynb
│   └── fine_tuning_best_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing_data.py
│   ├── training.py
│   └── app.py
│
├── model.pkl
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## FastAPI

The trained model is exposed through a REST API using FastAPI.

### Endpoints

| Method | Endpoint   | Description               |
| ------ | ---------- | ------------------------- |
| GET    | `/`        | Check API status          |
| POST   | `/predict` | Generate churn prediction |

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.7,
  "TotalCharges": 171.4
}
```

### Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.75
}
```

Where:

```text
0 = Predicted not to churn
1 = Predicted to churn
```

---

## Swagger Documentation

FastAPI provides interactive API documentation at:

```text
http://localhost:8000/docs
```

For the deployed API:

```text
https://customer-churn-project-a4xu.onrender.com/docs
```

---

## Run Locally

### Clone the repository

```bash
git clone https://github.com/Vineetyadav07-42/Customer-Churn-Project.git
cd Customer_Churn_Prediction_Resume
```

### Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python -m src.training
```

This generates `model.pkl`.

### Start the API

```bash
uvicorn src.app:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## Docker

### Build the image

```bash
docker build -t customer-churn-api .
```

### Run the container

```bash
docker run -p 8000:8000 customer-churn-api
```

The API will be available at:

```text
http://localhost:8000
```

---

## Deployment

The application is containerized using Docker and deployed on **Render**.

**Live API:** https://customer-churn-project-a4xu.onrender.com

**Swagger Documentation:** https://customer-churn-project-a4xu.onrender.com/docs

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Pydantic
* Uvicorn
* Joblib
* Docker
* Git & GitHub
* Render

---

## Author

**Vineet Yadav**

GitHub: Vineetyadav07-42
