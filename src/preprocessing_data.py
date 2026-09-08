from  pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer



numerical_columns = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

categorical_columns = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

preprocessing=make_column_transformer(
                            (StandardScaler(),numerical_columns),
                            (OneHotEncoder(handle_unknown='ignore'),categorical_columns)
                            )


