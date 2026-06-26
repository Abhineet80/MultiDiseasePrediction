from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

BASE_DIR = "ml_data"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

DATASET_CONFIG = {
    "diabetes": {"target": "Outcome"},
    "cardio": {"target": "target"},
    "kidney": {"target": "classification"},
    "hyper": {"target": "Has_Hypertension"}
}

# ================= TRAIN =================
@app.route('/train', methods=['POST'])
def train_models():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    dataset_type = request.form.get('disease_type')

    if dataset_type not in DATASET_CONFIG:
        return jsonify({"error": "Invalid disease type"}), 400

    file_path = os.path.join(BASE_DIR, file.filename)
    file.save(file_path)

    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    target_col = DATASET_CONFIG[dataset_type]["target"]

    if dataset_type == 'kidney':
        df[target_col] = df[target_col].astype(str).str.lower()
        df[target_col] = df[target_col].apply(lambda x: 1 if 'ckd' in x and 'not' not in x else 0)
        df.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
        df.replace(r'\t', '', regex=True, inplace=True)
    else:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].astype(str).str.lower()
            df[target_col] = df[target_col].apply(lambda x: 1 if x in ['yes', '1', 'true', 'ckd'] else 0)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df.dropna(subset=[target_col], inplace=True)

    y = df[target_col].astype(int)
    X = df.drop(target_col, axis=1)

    keywords = ['age', 'sex', 'gender', 'bmi', 'gluc', 'bgr', 'sugar', 'chol', 'bp', 'pressure', 'sc', 'creat', 'smok']
    cols_to_keep = [c for c in X.columns if any(k in c.lower() for k in keywords) or c.lower() == 'sc']
    X = X[cols_to_keep]

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).str.lower()
            if any(kw in col.lower() for kw in ['sex', 'gender']):
                X[col] = X[col].apply(lambda x: 1 if x == 'male' else 0)
            elif any(kw in col.lower() for kw in ['smok']):
                X[col] = X[col].apply(lambda x: 1 if x in ['yes', 'smoker'] else 0)
            elif 'bp_history' in col.lower():
                X[col] = X[col].apply(lambda x: 1 if 'high' in str(x).lower() or '1' in str(x) else 0)
            elif any(kw in col.lower() for kw in ['sugar', 'gluc']):
                X[col] = X[col].apply(lambda x: 1 if 'greater' in str(x).lower() or '1' in str(x) or 'yes' in str(x).lower() else 0)
            else:
                X[col] = pd.factorize(X[col])[0]
        else:
             X[col] = pd.to_numeric(X[col], errors='coerce')

    medians = X.median().to_dict()
    X.fillna(value=medians, inplace=True)
    joblib.dump(medians, f"models/{dataset_type}_medians.pkl")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')

    rf.fit(X_scaled, y)
    lr.fit(X_scaled, y)

    joblib.dump(scaler, f"models/{dataset_type}_scaler.pkl")
    joblib.dump(rf, f"models/{dataset_type}_rf.pkl")
    joblib.dump(lr, f"models/{dataset_type}_lr.pkl")
    joblib.dump(X.columns.tolist(), f"models/{dataset_type}_features.pkl")
    
    background_data = X_scaled[:150] if len(X_scaled) > 150 else X_scaled
    joblib.dump(background_data, f'models/{dataset_type}_background.pkl')

    return jsonify({"message": f"{dataset_type.capitalize()} trained successfully"})


# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        systolic_bp = float(data.get('bp', '120/80').split('/')[0])
    except:
        systolic_bp = 120

    patient = {
        'age': float(data.get('age', 0)),
        'bmi': float(data.get('bmi', 0)),
        'glucose': float(data.get('glucose', 0)),
        'chol': float(data.get('chol', 0)),
        'bp': systolic_bp,
        'gender': 1 if data.get('gender') == 'Male' else 0,
        'smoke': 1 if data.get('smoke') == 'Yes' else 0,
        'creat': float(data.get('creat', 1.0))
    }

    results = {}

    for disease in DATASET_CONFIG.keys():
        try:
            scaler = joblib.load(f"models/{disease}_scaler.pkl")
            rf = joblib.load(f"models/{disease}_rf.pkl")
            lr = joblib.load(f"models/{disease}_lr.pkl")
            features = joblib.load(f"models/{disease}_features.pkl")
            background_data = joblib.load(f"models/{disease}_background.pkl")

            row = []
            for f in features:
                f_low = f.lower()
                if 'age' in f_low: row.append(patient['age'])
                elif 'sex' in f_low or 'gender' in f_low: row.append(patient['gender'])
                elif 'bmi' in f_low: row.append(patient['bmi'])
                elif 'gluc' in f_low or 'bgr' in f_low: row.append(patient['glucose'])
                elif 'fasting_blood_sugar' in f_low: row.append(1.0 if patient['glucose'] >= 126 else 0.0)
                elif 'chol' in f_low: row.append(patient['chol'])
                elif 'bp' in f_low or 'pressure' in f_low or 'trestbps' in f_low: row.append(patient['bp'])
                elif 'bp_history' in f_low: row.append(1.0 if patient['bp'] > 130 else 0.0)
                elif 'sc' == f_low or 'creat' in f_low: row.append(patient['creat'])
                elif 'smok' in f_low: row.append(patient['smoke'])
                else: row.append(0)

            X = scaler.transform([row])

            rf_score = float(rf.predict_proba(X)[0][1])
            lr_score = float(lr.predict_proba(X)[0][1])

            # ================= THE CLINICAL RULE FIX =================

            if (patient['bp'] < 120 and patient['glucose'] < 100 and
                patient['chol'] < 180 and patient['creat'] < 1.2 and
                patient['bmi'] < 25 and patient['smoke'] == 0):
                rf_score = min(rf_score, 0.1)
                lr_score = min(lr_score, 0.1)

            if disease == 'diabetes':
                if patient['glucose'] < 110:
                    rf_score = min(rf_score, 0.25)
                    lr_score = min(lr_score, 0.25)
                elif patient['glucose'] < 160:
                    rf_score = min(rf_score, 0.65)
                    lr_score = min(lr_score, 0.65)
                    
            if disease == 'hyper':
                if patient['bp'] >= 140:
                    rf_score = max(rf_score, 0.72)
                    lr_score = max(lr_score, 0.72)
                elif patient['bp'] >= 130:
                    rf_score = max(rf_score, 0.45)
                    lr_score = max(lr_score, 0.45)
                elif patient['bp'] < 120:
                    rf_score = min(rf_score, 0.2)
                    lr_score = min(lr_score, 0.2)
                    
            if disease == 'cardio':
                if patient['bp'] < 140 and patient['chol'] < 220 and patient['smoke'] == 0:
                    rf_score = min(rf_score, 0.55)
                    lr_score = min(lr_score, 0.55)

            if disease == 'kidney':
                if patient['creat'] < 1.3:
                    rf_score = min(rf_score, 0.25)
                    lr_score = min(lr_score, 0.25)

            # Clamp
            rf_score = max(min(rf_score, 0.95), 0.02)
            lr_score = max(min(lr_score, 0.95), 0.02)

            # ================= EXPLAINABILITY =================
            clean_features = []
            for f in features:
                c = f.lower()
                if 'age' in c: clean_features.append("Age")
                elif 'sex' in c or 'gender' in c: clean_features.append("Gender")
                elif 'bmi' in c: clean_features.append("BMI")
                elif 'gluc' in c or 'bgr' in c or 'sugar' in c: clean_features.append("Glucose Level")
                elif 'chol' in c: clean_features.append("Cholesterol")
                elif 'bp' in c or 'pressure' in c: clean_features.append("Blood Pressure")
                elif 'sc' == c or 'creat' in c: clean_features.append("Creatinine")
                elif 'smok' in c: clean_features.append("Smoking Status")
                else: clean_features.append(f)

            shap_array = [0] * len(clean_features)
            lime_array = [0] * len(clean_features)

            try:
                import shap
                explainer_shap = shap.TreeExplainer(rf)
                shap_vals = explainer_shap.shap_values(X)
                if isinstance(shap_vals, list): shap_array = shap_vals[1][0].tolist()
                elif len(shap_vals.shape) == 3: shap_array = shap_vals[0, :, 1].tolist()
                else: shap_array = shap_vals[0].tolist()

                import lime
                import lime.lime_tabular
                explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                    training_data=background_data,
                    feature_names=clean_features,
                    class_names=['Negative', 'Positive'],
                    mode='classification',
                    random_state=42
                )
                exp = explainer_lime.explain_instance(data_row=X[0], predict_fn=rf.predict_proba)
                lime_weights_dict = dict(exp.local_exp[1]) 
                lime_array = [lime_weights_dict.get(i, 0.0) for i in range(len(clean_features))]
                
                # De-bias Age/Gender in charts so actionable features are highly visible
                for idx, feat in enumerate(clean_features):
                    if feat in ['Age', 'Gender']:
                        shap_array[idx] = shap_array[idx] * 0.4
                        lime_array[idx] = lime_array[idx] * 0.4
                        
            except Exception as e:
                pass

            results[disease] = {
                "rf": rf_score,
                "lr": lr_score,
                "explain": {
                    "features": clean_features,
                    "shap": shap_array,
                    "lime": lime_array
                }
            }

        except Exception as e:
            print(f"Error processing {disease}: {e}")
            continue

    if not results:
         return jsonify({"error": "No models are trained yet."}), 400

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)