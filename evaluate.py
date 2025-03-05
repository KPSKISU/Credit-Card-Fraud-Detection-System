import joblib
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess_data

# Load model
model = joblib.load("../models/fraud_model.pkl")

# Load data
_, X_test, _, y_test = load_and_preprocess_data("../data/creditcard.csv")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
