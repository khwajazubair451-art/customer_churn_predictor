
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("telco_churn.csv")

# 2. Clean data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)

# 3. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 5. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# 6. Create complete machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ))
])

# 7. Train model
model.fit(X, y)

# 8. Save model
joblib.dump(model, "churn_model.pkl")

print("Model trained and saved successfully!")
print("Scikit-learn version:", __import__("sklearn").__version__)
