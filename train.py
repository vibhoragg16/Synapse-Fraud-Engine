import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import shap
import warnings

warnings.filterwarnings("ignore")

print("Starting the ultimate training and comparison process...")

# --- 1. Data Loading and Initial Split ---
df = pd.read_csv('creditcard.csv')
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. Model Definitions ---
def create_nn_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def create_ensemble_model():
    clf1 = LogisticRegression(random_state=42, solver='liblinear')
    clf2 = RandomForestClassifier(random_state=42, n_estimators=50)
    clf3 = lgb.LGBMClassifier(random_state=42)
    return VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('lgbm', clf3)], voting='soft')

# --- 3. Run Experiments ---
samplers = {'Undersampling': RandomUnderSampler(random_state=42), 'SMOTE': SMOTE(random_state=42)}
results = {}

for sampler_name, sampler in samplers.items():
    print(f"\n--- Running experiment with {sampler_name} ---")
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Train Ensemble
    print(f"Training Ensemble model...")
    ensemble = create_ensemble_model().fit(X_resampled, y_resampled)
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    results[f'Ensemble_{sampler_name}'] = {'model': ensemble, 'proba': ensemble_proba, 'score': average_precision_score(y_test, ensemble_proba)}
    print(f"Ensemble AUPRC: {results[f'Ensemble_{sampler_name}']['score']:.4f}")

    # Train Neural Network
    print(f"Training Neural Network...")
    nn = create_nn_model(X_resampled.shape[-1])
    nn.fit(X_resampled, y_resampled, epochs=10, batch_size=2048, verbose=0)
    nn_proba = nn.predict(X_test, verbose=0).flatten()
    results[f'NN_{sampler_name}'] = {'model': nn, 'proba': nn_proba, 'score': average_precision_score(y_test, nn_proba)}
    print(f"NN AUPRC: {results[f'NN_{sampler_name}']['score']:.4f}")

# --- 4. Visualize All Results ---
print("\nGenerating comparison plot...")
plt.figure(figsize=(12, 8))
for name, data in results.items():
    precision, recall, _ = precision_recall_curve(y_test, data['proba'])
    plt.plot(recall, precision, label=f"{name} (AUPRC = {data['score']:.4f})")
plt.title('Model Comparison on Different Sampling Strategies')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison_pr_curve.png')
print("Saved 'model_comparison_pr_curve.png'")


best_model_name = max(results, key=lambda name: results[name]['score'])
best_model_info = results[best_model_name]
best_model = best_model_info['model']

print(f"\n--- Best Model Selected: {best_model_name} with AUPRC = {best_model_info['score']:.4f} ---")

joblib.dump(scaler, 'ultimate_scaler.pkl')
print("Saved scaler to 'ultimate_scaler.pkl'")

# Save the winning model
if 'NN' in best_model_name:
    best_model.save('ultimate_model.h5')
    print("Saved winning Keras model to 'ultimate_model.h5'")
else:
    joblib.dump(best_model, 'ultimate_model.pkl')
    print("Saved winning scikit-learn model to 'ultimate_model.pkl'")


print("Creating and saving SHAP explainer...")

winning_sampler = samplers[best_model_name.split('_')[1]]
X_background, _ = winning_sampler.fit_resample(X_train, y_train)


if 'NN' in best_model_name:
    
    def predict_fn(x):
        return best_model.predict(x)
else:
    
    def predict_fn(x):
        return best_model.predict_proba(x)

explainer = shap.KernelExplainer(predict_fn, shap.sample(X_background, 100))
joblib.dump(explainer, 'ultimate_shap_explainer.pkl')
print("Saved SHAP explainer to 'ultimate_shap_explainer.pkl'")

print("\nUltimate training complete!")
