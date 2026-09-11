from src.preprocessing_data import preprocessing
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import make_pipeline

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=BASE_DIR/'Data'/'customerchurn.csv'
MODEL_PATH=BASE_DIR/'model.pkl'

data=pd.read_csv(DATA_PATH)


data['TotalCharges'] = data['TotalCharges'].replace(' ', '0')
data['TotalCharges'] = data['TotalCharges'].astype(float)


data['Churn'] = data['Churn'].map({
    'No': 0,
    'Yes': 1
})



data=data.drop(columns=['customerID'])

X=data.drop(columns=['Churn'])
y=data['Churn']


final_pipeline=make_pipeline(
                             preprocessing,
                             LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=1000)
                             )



final_pipeline.fit(X,y)

joblib.dump(final_pipeline,MODEL_PATH)

print(f'Model trained and saved to: {MODEL_PATH}')