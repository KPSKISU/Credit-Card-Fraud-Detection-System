import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna()

    # Separate features and labels
    X = df.drop(columns=['Class'])  # Assuming 'Class' is the target
    y = df['Class']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    filepath = "../data/creditcard.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    print("Data preprocessing completed.")
