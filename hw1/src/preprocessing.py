
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess_data(df):

    # 1. Split target
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # 2. Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=42
    )

    # 3. Missing values
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_fill_values = {}
    categorical_fill_values = {}

    for col in num_cols:
        median = X_train[col].median()
        numeric_fill_values[col] = median
        X_train[col] = X_train[col].fillna(median)
        X_val[col] = X_val[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in cat_cols:
        mode = X_train[col].mode()[0]
        categorical_fill_values[col] = mode
        X_train[col] = X_train[col].fillna(mode)
        X_val[col] = X_val[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)

    # 4. Encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 5. Feature Engineering
    X_train["Engagement"] = X_train["ProductRelated"] + X_train["Informational"]
    X_val["Engagement"] = X_val["ProductRelated"] + X_val["Informational"]
    X_test["Engagement"] = X_test["ProductRelated"] + X_test["Informational"]

    X_train["Duration_per_page"] = X_train["ProductRelated_Duration"] / (X_train["ProductRelated"] + 1)
    X_val["Duration_per_page"] = X_val["ProductRelated_Duration"] / (X_val["ProductRelated"] + 1)
    X_test["Duration_per_page"] = X_test["ProductRelated_Duration"] / (X_test["ProductRelated"] + 1)

    # Μετά το feature engineering ξανακρατάμε τα τελικά feature names
    feature_columns = X_train.columns.tolist()

    # Ευθυγράμμιση ξανά, για ασφάλεια
    X_val = X_val.reindex(columns=feature_columns, fill_value=0)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 7. Save scaler + metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    preprocessing_metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "feature_columns": feature_columns
    }

    joblib.dump(preprocessing_metadata, "models/preprocessing_metadata.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

