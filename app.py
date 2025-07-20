from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from tensorflow import keras
import os
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Ultimate Fraud Detection API",
    description="Serves the best model from a comprehensive experiment, with SHAP explainability."
)

# --- Load All Artifacts ---
model, scaler, explainer, model_type = None, None, None, None
try:
    if os.path.exists('ultimate_model.h5'):
        model = keras.models.load_model('ultimate_model.h5')
        model_type = 'keras'
    elif os.path.exists('ultimate_model.pkl'):
        model = joblib.load('ultimate_model.pkl')
        model_type = 'sklearn'
    
    scaler = joblib.load('ultimate_scaler.pkl')
    explainer = joblib.load('ultimate_shap_explainer.pkl')
    print(f"Artifacts loaded successfully. Winning model type: {model_type}")
except FileNotFoundError:
    print("Error: Artifacts not found. Please run train_ultimate.py first.")

class Transaction(BaseModel):
    Time: float; V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ultimate Fraud Detection API."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if not all([model, scaler, explainer]):
        return {"error": "Artifacts not loaded. Cannot make predictions."}

    input_data = pd.DataFrame([transaction.dict()])
    if 'Class' in input_data.columns:
        input_data.drop('Class', axis=1, inplace=True)
        
    input_data['scaled_amount'] = scaler.transform(input_data['Amount'].values.reshape(-1, 1))
    input_data['scaled_time'] = scaler.transform(input_data['Time'].values.reshape(-1, 1))
    input_data.drop(['Time', 'Amount'], axis=1, inplace=True)

    # --- Prediction based on model type ---
    if model_type == 'keras':
        probability = model.predict(input_data, verbose=0)[0][0]
    else: # sklearn
        probability = model.predict_proba(input_data)[0, 1]
    
    is_fraud = bool(probability > 0.5)
    response = {"is_fraud": is_fraud, "fraud_probability": f"{probability:.4f}"}

    if is_fraud:
        print("Fraud detected, generating SHAP explanation...")
        shap_values = explainer.shap_values(input_data)
        
        shap_values_fraud = shap_values[1][0]
        
        feature_importance = pd.Series(shap_values_fraud, index=input_data.columns)
        top_features = feature_importance.reindex(feature_importance.abs().sort_values(ascending=False).index).head(3)
        
        response["explanation"] = {
            "message": "Top 3 features contributing to this fraud score:",
            "features": top_features.to_dict()
        }
    return response
