from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Online Shoppers Prediction API")

# Load model and preprocessing artifacts
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
metadata = joblib.load("models/preprocessing_metadata.pkl")


class ShopperInput(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


def preprocess_single_input(data: ShopperInput):
    # 1. raw dict -> DataFrame
    df = pd.DataFrame([data.model_dump()])

    # 2. missing values
    for col, fill_value in metadata["numeric_fill_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    for col, fill_value in metadata["categorical_fill_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # 3. encoding
    df = pd.get_dummies(df, drop_first=True)

    # 4. add engineered features
    if "ProductRelated" in df.columns and "Informational" in df.columns:
        df["Engagement"] = df["ProductRelated"] + df["Informational"]
    else:
        df["Engagement"] = 0

    if "ProductRelated_Duration" in df.columns and "ProductRelated" in df.columns:
        df["Duration_per_page"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    else:
        df["Duration_per_page"] = 0

    # 5. align columns with training
    df = df.reindex(columns=metadata["feature_columns"], fill_value=0)

    # 6. scale
    X_scaled = scaler.transform(df)

    return X_scaled


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: ShopperInput):
    X_processed = preprocess_single_input(data)

    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    label = "Purchase" if prediction == 1 else "No Purchase"

    return {
        "prediction": int(prediction),
        "label": label,
        "probability": float(probability)
    }