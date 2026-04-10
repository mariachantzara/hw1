from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/classical_model.pkl")

    return model