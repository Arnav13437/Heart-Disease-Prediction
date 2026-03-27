import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save():
    print("Loading dataset...")
    # Load dataset
    csv_path = 'heart_disease_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    heart_data = pd.read_csv(csv_path)
    X = heart_data.drop(columns=['target'])
    Y = heart_data['target']

    print("Training Logistic Regression model...")
    # Train model on all data for maximum accuracy in production
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)

    print("Saving model to model.joblib...")
    joblib.dump(model, 'model.joblib')
    print("Model saved successfully. You are ready to deploy!")

if __name__ == "__main__":
    train_and_save()
