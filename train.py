import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/creditcard.csv")

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/fraud_model.pkl")
print("Model training completed and saved.")
