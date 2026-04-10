import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from src.preprocessing import preprocess_data
from src.train_classical import train_random_forest
from src.train_neural import train_neural_network
from src.evaluate import evaluate_random_forest, evaluate_neural_network, compare_models
#from train_neural import train_neural_network_tanh

df = pd.read_csv("dataset.csv")

# Correlation matrix πριν το encoding / feature engineering
X_raw = df.drop("Revenue", axis=1)

# κρατάμε μόνο αριθμητικές στήλες για correlation
X_raw_numeric = X_raw.select_dtypes(include=["int64", "float64", "bool"])

corr = X_raw_numeric.corr()

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

print("Preprocessing complete!")
print(X_train.shape, X_val.shape, X_test.shape)

# --------------------------------
# PCA μετά το scaling
# --------------------------------
pca = PCA()
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

# PCA με 2 components για scatter plot και loadings
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_train)

# 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="coolwarm", alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.colorbar(label="Revenue")
plt.show()


feature_names = pd.get_dummies(
    df.drop("Revenue", axis=1), drop_first=True
).columns.tolist()

feature_names = feature_names + ["Engagement", "Duration_per_page"]

loadings = pd.DataFrame(
    pca2.components_.T,
    columns=["PC1", "PC2"],
    index=feature_names
)

print("PCA Loadings:")
print(loadings.sort_values(by="PC1", key=abs, ascending=False))

top_loadings = loadings["PC1"].abs().sort_values(ascending=False).head(8)
print("top_loadings:", top_loadings)

# Train model
model = train_random_forest(X_train, y_train)

print("Training Neural Network...")

nn_model, history = train_neural_network(
    X_train, y_train,
    X_val, y_val
)

print("Neural network training complete!")

#Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


#Evaluation
rf_results = evaluate_random_forest(model, X_test, y_test)
nn_results = evaluate_neural_network(nn_model, X_test, y_test)

comparison_df = compare_models(rf_results, nn_results)

#Hyperparameter tuning
print("\n Hyperparameter Tuning")

param_dist = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 7, 10],
    "min_samples_leaf": [1, 2, 3, 4]
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best parameters:", search.best_params_)

# validation check
y_val_proba = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)

print("Validation ROC-AUC after tuning:", val_auc)

y_test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)

print("Test ROC-AUC after tuning:", test_auc)