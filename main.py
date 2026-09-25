from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "loan_default_final_model.pkl"
SCALER_PATH = MODEL_DIR / "loan_default_final_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "loan_default_final_features.pkl"
THRESHOLD_PATH = MODEL_DIR / "loan_default_final_threshold.pkl"

HISTORY_PATH = BASE_DIR / "history.json"


# ============================================================
# 2. PERSISTENT PREDICTION HISTORY
# ============================================================

def load_prediction_history():

    if not HISTORY_PATH.exists():
        return []

    try:
        with open(HISTORY_PATH, "r") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


def save_prediction_history(history):

    with open(HISTORY_PATH, "w") as file:
        json.dump(
            history,
            file,
            indent=4
        )


prediction_history = load_prediction_history()


# ============================================================
# 3. FINAL MODEL EVALUATION METRICS
# ============================================================

MODEL_METRICS = {
    "accuracy": 0.793538,
    "precision": 0.285222,
    "recall": 0.516439,
    "f1Score": 0.367487,
    "rocAuc": 0.754406,
    "threshold": 0.55
}


# ============================================================
# 4. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="LoanGuard API",
    description="Loan Default Prediction API",
    version="1.0.0"
)


# ============================================================
# 5. CORS
# ============================================================

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://ml-frontend-pi.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# 6. CHECK MODEL FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Final model file not found: {MODEL_PATH}"
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Final scaler file not found: {SCALER_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Final features file not found: {FEATURES_PATH}"
    )

if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(
        f"Final threshold file not found: {THRESHOLD_PATH}"
    )


# ============================================================
# 7. LOAD FINAL MODEL, SCALER, FEATURES AND THRESHOLD
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
classification_threshold = joblib.load(THRESHOLD_PATH)


# ============================================================
# 8. CONVERT FEATURES INTO LIST
# ============================================================

if isinstance(features, pd.DataFrame):

    feature_names = features.columns.tolist()

elif isinstance(features, pd.Series):

    feature_names = features.tolist()

else:

    feature_names = list(features)


# ============================================================
# 9. STARTUP INFORMATION
# ============================================================

print("========================================")
print("LoanGuard API")
print("========================================")
print("Final Tuned Random Forest loaded successfully.")
print("Scaler loaded successfully.")
print("Features loaded successfully.")
print("Threshold loaded successfully.")
print("Number of features:", len(feature_names))
print("Classification threshold:", classification_threshold)
print("Saved prediction history:", len(prediction_history))
print("Allowed CORS origins:", ALLOWED_ORIGINS)
print("========================================")


# ============================================================
# 10. NUMERICAL COLUMNS
# ============================================================

numerical_columns = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]


# ============================================================
# 11. CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "LoanPurpose"
]


# ============================================================
# 12. BINARY COLUMNS
# ============================================================

binary_columns = [
    "HasMortgage",
    "HasDependents",
    "HasCoSigner"
]


# ============================================================
# 13. REQUEST BODY
# ============================================================

class LoanApplication(BaseModel):

    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str

    HasMortgage: bool
    HasDependents: bool

    LoanPurpose: str

    HasCoSigner: bool


# ============================================================
# 14. ROOT ROUTE
# ============================================================

@app.get("/")
def root():

    return {
        "message": "LoanGuard API is running!",
        "model": "Tuned Random Forest",
        "threshold": float(classification_threshold),
        "features": len(feature_names),
        "history_records": len(prediction_history),
        "cors": "enabled"
    }


# ============================================================
# 15. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "LoanGuard backend is running",
        "model": "Tuned Random Forest",
        "cors": "enabled"
    }


# ============================================================
# 16. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_loan(application: LoanApplication):

    try:

        # Convert Pydantic object to dictionary
        data = application.model_dump()

        # Create DataFrame
        df = pd.DataFrame([data])

        # Convert boolean columns to 0 / 1
        for column in binary_columns:
            df[column] = df[column].astype(int)

        # One-hot encoding
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
            dtype=int
        )

        # Match training features
        df = df.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale numerical columns
        df[numerical_columns] = scaler.transform(
            df[numerical_columns]
        )

        # Model probability
        probability = float(
            model.predict_proba(df)[0][1]
        )

        # Apply final threshold
        prediction = int(
            probability >= classification_threshold
        )

        # Risk level
        if probability < 0.30:
            risk_level = "Low"

        elif probability < 0.60:
            risk_level = "Moderate"

        else:
            risk_level = "High"

        # Feature importance
        feature_importances = model.feature_importances_

        top_indices = feature_importances.argsort()[::-1][:8]

        model_features = []

        for index in top_indices:

            model_features.append({
                "feature": feature_names[index],
                "importance": round(
                    float(feature_importances[index]),
                    4
                )
            })

        # Create history record
        history_record = {
            "loanAmount": application.LoanAmount,
            "creditScore": application.CreditScore,
            "defaultProbability": round(
                probability,
                4
            ),
            "prediction": prediction,
            "riskLevel": risk_level
        }

        # Add prediction to history
        prediction_history.append(history_record)

        # Save history
        save_prediction_history(prediction_history)

        # Response
        return {
            "prediction": prediction,

            "default_probability": round(
                probability,
                4
            ),

            "risk_level": risk_level,

            "model": "Tuned Random Forest",

            "threshold": float(
                classification_threshold
            ),

            "factors": {
                "modelFeatures": model_features
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# 17. PREDICTION HISTORY
# ============================================================

@app.get("/history")
def get_prediction_history():

    return prediction_history


# ============================================================
# 18. RISK PROBABILITY DISTRIBUTION
# ============================================================

def get_risk_probability_distribution():

    buckets = [
        {"bucket": "0-10%", "count": 0},
        {"bucket": "10-20%", "count": 0},
        {"bucket": "20-30%", "count": 0},
        {"bucket": "30-40%", "count": 0},
        {"bucket": "40-50%", "count": 0},
        {"bucket": "50-60%", "count": 0},
        {"bucket": "60-70%", "count": 0},
        {"bucket": "70-80%", "count": 0},
        {"bucket": "80-90%", "count": 0},
        {"bucket": "90-100%", "count": 0}
    ]

    for item in prediction_history:

        probability = float(
            item["defaultProbability"]
        )

        index = min(
            int(probability * 10),
            9
        )

        buckets[index]["count"] += 1

    return buckets


# ============================================================
# 19. FEATURE OVERVIEW
# ============================================================

def get_feature_overview():

    try:

        feature_importances = model.feature_importances_

        feature_importance = []

        for feature, importance in zip(
            feature_names,
            feature_importances
        ):

            feature_importance.append({
                "feature": feature,
                "importance": float(importance)
            })

        feature_importance.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        top_features = feature_importance[:8]

        total_importance = sum(
            item["importance"]
            for item in top_features
        )

        if total_importance > 0:

            for item in top_features:

                item["importance"] = round(
                    item["importance"]
                    / total_importance,
                    4
                )

        # User-friendly feature names
        for item in top_features:

            feature = item["feature"]

            if feature.startswith("EmploymentType_"):

                feature = (
                    "Employment Type: "
                    + feature.replace(
                        "EmploymentType_",
                        ""
                    )
                )

            elif feature.startswith("Education_"):

                feature = (
                    "Education: "
                    + feature.replace(
                        "Education_",
                        ""
                    )
                )

            elif feature.startswith("MaritalStatus_"):

                feature = (
                    "Marital Status: "
                    + feature.replace(
                        "MaritalStatus_",
                        ""
                    )
                )

            elif feature.startswith("LoanPurpose_"):

                feature = (
                    "Loan Purpose: "
                    + feature.replace(
                        "LoanPurpose_",
                        ""
                    )
                )

            elif feature == "HasMortgage":

                feature = "Has Mortgage"

            elif feature == "HasDependents":

                feature = "Has Dependents"

            elif feature == "HasCoSigner":

                feature = "Has Co-Signer"

            item["feature"] = feature

        return top_features

    except Exception as e:

        print(
            "Feature overview error:",
            str(e)
        )

        return []


# ============================================================
# 20. ANALYTICS
# ============================================================

@app.get("/analytics")
def get_analytics():

    total_applications = len(
        prediction_history
    )

    # Default / No Default
    default_predictions = sum(
        item["prediction"] == 1
        for item in prediction_history
    )

    no_default_predictions = sum(
        item["prediction"] == 0
        for item in prediction_history
    )

    # Average risk probability
    if total_applications > 0:

        average_risk_probability = (
            sum(
                item["defaultProbability"]
                for item in prediction_history
            )
            / total_applications
        )

    else:

        average_risk_probability = 0

    # Risk distribution
    low_risk = sum(
        item["riskLevel"] == "Low"
        for item in prediction_history
    )

    moderate_risk = sum(
        item["riskLevel"] == "Moderate"
        for item in prediction_history
    )

    high_risk = sum(
        item["riskLevel"] == "High"
        for item in prediction_history
    )

    # Probability distribution
    risk_probability_distribution = (
        get_risk_probability_distribution()
    )

    # Feature importance
    feature_overview = (
        get_feature_overview()
    )

    return {

        "totals": {

            "totalApplications":
                total_applications,

            "defaultPredictions":
                default_predictions,

            "noDefaultPredictions":
                no_default_predictions,

            "averageRiskProbability":
                round(
                    average_risk_probability,
                    4
                )
        },

        "defaultVsNoDefault": [

            {
                "name": "No Default",
                "value": no_default_predictions
            },

            {
                "name": "Default",
                "value": default_predictions
            }
        ],

        "riskDistribution": [

            {
                "name": "Low Risk",
                "value": low_risk
            },

            {
                "name": "Moderate Risk",
                "value": moderate_risk
            },

            {
                "name": "High Risk",
                "value": high_risk
            }
        ],

        "riskProbabilityDistribution":
            risk_probability_distribution,

        "featureOverview":
            feature_overview,

        "model": {

            "name":
                "Tuned Random Forest",

            "status":
                "Evaluated",

            "accuracy":
                MODEL_METRICS["accuracy"],

            "precision":
                MODEL_METRICS["precision"],

            "recall":
                MODEL_METRICS["recall"],

            "f1Score":
                MODEL_METRICS["f1Score"],

            "rocAuc":
                MODEL_METRICS["rocAuc"],

            "threshold":
                MODEL_METRICS["threshold"]
        }
    }