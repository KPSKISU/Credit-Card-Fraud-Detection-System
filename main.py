from src.preprocess import load_and_preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_fraud

if __name__ == "__main__":
    print("Starting Credit Card Fraud Detection System...")

    # Data Preparation
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")

    # Train Model
    train_model(X_train, y_train)

    # Evaluate Model
    evaluate_model(X_test, y_test)

    # Test a sample transaction
    sample_transaction = [0.2, -1.3, 3.4, ..., 0.8]  # Replace with actual feature values
    print("Transaction Prediction:", predict_fraud(sample_transaction))
