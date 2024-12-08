import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

from datetime import datetime

from data_preproces_piplines.data_process import data_preprocess

# Define prediction steps
predict_future_step = 2

# Preprocess data function
def preprocess_data():
    # Load and preprocess the data
    df = data_preprocess()
    df = df[df['GEO'] == 'Canada']

    continuous_columns = [
        'Participation Rate', 
        'Population', 
        'CPI', 
        'Gross domestic product at market prices', 
        'Gross fixed capital formation', 
        'Minimum Wage'
    ]

    # Convert REF_DATE to datetime and extract month
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df['month'] = df['REF_DATE'].dt.month

    # Create cyclical monthly features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    date_features = ['month_sin', 'month_cos']
    
    # Scale continuous features using RobustScaler
    scaler_X = RobustScaler()
    continuous_features = df[continuous_columns].astype(float)
    scaler_X.fit(continuous_features)
    scaled_features = scaler_X.transform(continuous_features)
    
    # Scale the target (Unemployment Rate)
    scaler_Y = RobustScaler()
    label = df['Unemployment Rate'].astype(float).values.reshape(-1, 1)
    scaler_Y.fit(label)
    scaled_label = scaler_Y.transform(label)
    
    # Combine scaled features with date_features
    features = np.hstack((scaled_features, df[date_features].values))
    label = scaled_label
    
    # Define feature names
    feature_names = continuous_columns + date_features
    
    # Split into training and test sets (Shift by predict_future_step)
    X_train = features[:-predict_future_step]
    y_train = label[:-predict_future_step]
    X_test = features[-predict_future_step:]
    y_test = label[-predict_future_step:]
    
    return X_train, X_test, y_train, y_test, scaler_Y, df, feature_names

# Linear Regression Model class
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def coefficients(self):
        return self.model.coef_
    
    def intercept(self):
        return self.model.intercept_

# Main function
def main():
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_Y, df, feature_names = preprocess_data()
    
    # Print dataset shapes
    print("\n" + "-"*50)
    print("Data Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("-"*50 + "\n")
    
    # Initialize and train the Linear Regression model
    model = LinearRegressionModel()
    print("Training the Linear Regression model...\n")
    model.train(X_train, y_train)
    
    # Retrieve and print coefficients
    coefficients = model.coefficients()
    intercept = model.intercept()
    
    print("Model Coefficients:")
    print(f"Intercept: {intercept[0]:.4f}")
    print("Feature Coefficients:")
    for feature, coef in zip(feature_names, coefficients[0]):
        print(f"  {feature}: {coef:.4f}")
    print("-"*50 + "\n")
    
    # Predict on the test set
    forecast = model.predict(X_test)
    
    # Inverse transform predictions and actual test values
    y_pred_future = scaler_Y.inverse_transform(forecast)[:, 0]
    y_test_actual = scaler_Y.inverse_transform(y_test)[:, 0]
    
    # Retrieve the corresponding future dates from the dataframe
    future_dates = df['REF_DATE'].iloc[-predict_future_step:].values
    
    # Print predictions and actual values with corresponding dates
    print("\n" + "-"*50)
    print("Future Predictions vs Actual (by Date):")
    for i, date in enumerate(future_dates):
        formatted_date = datetime.strptime(str(date)[:10], "%Y-%m-%d").strftime("%B %Y")
        print(f"Date: {formatted_date} | Predicted = {y_pred_future[i]:.2f}, Actual = {y_test_actual[i]:.2f}")
    print("-"*50 + "\n")
    
    # Evaluate the model
    mae = mean_absolute_error(y_test_actual, y_pred_future)
    mse = mean_squared_error(y_test_actual, y_pred_future)
    print("Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error:  {mse:.4f}")
    print("-"*50 + "\n")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, y_test_actual, marker='o', label='Actual')
    plt.plot(future_dates, y_pred_future, marker='x', label='Predicted')
    plt.title('Actual vs Predicted Unemployment Rate')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
