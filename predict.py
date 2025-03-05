import joblib
import numpy as np
from preprocess import load_and_preprocess_data

def predict_fraud(transaction):
    # Load trained model
    model = joblib.load("../models/fraud_model.pkl")
    
    # Convert transaction to numpy array and reshape
    transaction = np.array(transaction).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(transaction)
    
    return "Fraudulent" if prediction[0] == 1 else "Legitimate"

if __name__ == "__main__":
    sample_transaction = [0.1, -2.3, 1.5, ..., 0.5]  # Replace with real feature values
    print("Prediction:", predict_fraud(sample_transaction))
