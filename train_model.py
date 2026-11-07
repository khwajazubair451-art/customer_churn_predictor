
from pathlib import Path
import pandas as pd
import joblib
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("telco_churn.csv")

# Clean dataset
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# Build pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ))
])

# Train model
model.fit(X, y)

# Save model in the exact project folder
model_path = Path(__file__).resolve().parent / "churn_model.pkl"

joblib.dump(model, model_path)

# Immediately test that the saved model can be loaded
test_model = joblib.load(model_path)

print("Model trained and saved successfully!")
print("Model loaded successfully!")
print("Scikit-learn version:", sklearn.__version__)
print("Saved location:", model_path)
