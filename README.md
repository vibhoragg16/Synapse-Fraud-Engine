# Synapse Fraud Engine: Real-time, Explainable Fraud Detection

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20TensorFlow%20%7C%20SHAP-orange.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

## 1. Overview

**Synapse Fraud Engine** is an advanced system designed to detect fraudulent credit card transactions in real-time. Unlike traditional models that only provide a prediction, Synapse leverages **Explainable AI (XAI)** to detail *why* a transaction is flagged, providing crucial transparency and building trust in the system.

The project follows a rigorous machine learning workflow:
1.  **Experimentation:** Systematically trains and evaluates multiple models (Ensemble Classifiers vs. Neural Networks) on different data sampling strategies (Undersampling vs. SMOTE).
2.  **Visualization:** Generates clear performance graphs to visually identify the champion model.
3.  **Deployment:** Serves the winning model through a high-performance FastAPI endpoint, complete with model explanations.

---

## 2. Project Architecture

The project is designed with a clear separation between the offline training pipeline and the online prediction service.

```
+--------------------------+
|   Raw Data               |
|   (creditcard.csv)       |
+--------------------------+
            |
            v
+-------------------------------------------------+
|   Training & Experimentation Pipeline           |
|   (train_ultimate.py)                           |
|-------------------------------------------------|
| 1. Preprocess Data (RobustScaler)               |
| 2. Run 4 Experiments:                           |
|    - Ensemble + Undersampling                   |
|    - Neural Network + Undersampling             |
|    - Ensemble + SMOTE                           |
|    - Neural Network + SMOTE                     |
| 3. Visualize Results & Select Best Model (AUPRC)|
+-------------------------------------------------+
            |
            | (Generates)
            v
+--------------------------+     +-----------------------------------+
|   Performance Graph      |     |   Saved Artifacts                 |
|   (comparison_curve.png) |     |   - ultimate_model.pkl / .h5      |
+--------------------------+     |   - ultimate_scaler.pkl           |
                                 |   - ultimate_shap_explainer.pkl   |
                                 +-----------------------------------+
                                                  |
                                                  | (Loaded by API)
                                                  v
+-------------------------------------------------+
|   Real-time Prediction Service                  |
|   (app_ultimate.py)                             |
|-------------------------------------------------|
| 1. Load Saved Artifacts                         |
| 2. Listen for POST requests on /predict         |
| 3. Preprocess incoming JSON data                |
| 4. Predict using the best model                 |
| 5. If fraud, generate SHAP explanation          |
| 6. Return JSON response with prediction &       |
|    explanation                                  |
+-------------------------------------------------+
            ^                                     |
            | (HTTP Request)                      | (HTTP Response)
            v                                     |
+--------------------------+                      |
|   User / Client App      | <--------------------+
+--------------------------+

```

---
You can download the data from here - (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
## 3. Key Features

* **Rigorous Model Comparison:** Automatically tests four distinct modeling strategies to empirically determine the most effective approach for the dataset.
* **Visual Performance Analysis:** Generates a comprehensive Precision-Recall Curve plot to visualize the performance of all candidate models.
* **Explainable AI (XAI):** For every fraudulent transaction detected, the API returns the top 3 features that contributed to the decision, powered by the SHAP library.
* **High-Performance API:** Built with FastAPI for fast, asynchronous, and production-ready prediction serving.
* **Automated Best Model Selection:** The training pipeline automatically selects and saves the model with the highest Area Under the Precision-Recall Curve (AUPRC) score.

---

## 4. Model Performance Visualization

The core of this project is its data-driven approach to model selection. The training script generates the following comparison graph, which clearly shows the superiority of the **Ensemble model trained on SMOTE data** for this specific problem. The AUPRC is the most important metric here due to the severe class imbalance in the dataset.

*(Note: You will need to add the actual image to your GitHub repository for it to display correctly)*
`![Model Comparison Chart](model_comparison_pr_curve.png)`

---

## 5. Technology Stack

* **Backend Framework:** FastAPI
* **Machine Learning:** Scikit-learn, TensorFlow/Keras, LightGBM
* **Data Handling:** Pandas, NumPy
* **Data Imbalance:** Imbalanced-learn (for SMOTE & Undersampling)
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Visualization:** Matplotlib, Seaborn
* **API Server:** Uvicorn

---

## 6. Setup and Usage

Follow these steps to run the project locally.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/vibhoragg16/Synapse-Fraud-Engine.git](https://github.com/vibhoragg16/Synapse-Fraud-Engine.git)
cd Synapse-Fraud-Engine
```

### Step 2: Install Dependencies
It is recommended to use a virtual environment.
```bash
# Create a requirements.txt file first if you don't have one:
# pip freeze > requirements.txt

pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline
This command will run the full experiment, generate the comparison graph (`model_comparison_pr_curve.png`), and save the best-performing model and its artifacts.
```bash
python train.py
```

### Step 4: Start the API Server
Once the training is complete, start the real-time prediction server.
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

---

## 7. API Endpoints

You can access the interactive API documentation (powered by Swagger UI) by navigating to `http://127.0.0.1:8000/docs`.

### `/predict`

This endpoint accepts a `POST` request with a JSON object representing a single transaction and returns a fraud prediction.

**Example `curl` command for a fraudulent transaction:**
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "Time": 472, "V1": -3.04, "V2": -3.15, "V3": 1.08, "V4": 2.28, "V5": 1.35, "V6": -1.06, "V7": 0.32, "V8": -0.06, "V9": -0.27, "V10": -0.83, "V11": -0.41, "V12": -0.5, "V13": -0.11, "V14": -2.2, "V15": 1.47, "V16": -2.2, "V17": -1.5, "V18": 0.66, "V19": 0.45, "V20": 2.1, "V21": 0.66, "V22": 0.43, "V23": 1.37, "V24": -0.29, "V25": -0.14, "V26": -0.25, "V27": 0.03, "V28": 0.03, "Amount": 529
}'
```

**Example Response (with Explanation):**
```json
{
  "is_fraud": true,
  "fraud_probability": "0.9876",
  "explanation": {
    "message": "Top 3 features contributing to this fraud score:",
    "features": {
      "V14": -0.75,
      "V4": 0.62,
      "V12": -0.55
    }
  }
}
```

---

## 8. Future Improvements

* **Real-time Data Stream:** Integrate the system with a message broker like Apache Kafka to process a continuous stream of live transactions.
* **Cloud Deployment:** Containerize the application with Docker and deploy it to a cloud service (e.g., AWS, GCP, Azure) for scalability.
* **Database Integration:** Log all incoming predictions and SHAP explanations to a database for monitoring and auditing purposes.
