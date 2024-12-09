import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from data_preproces_piplines.data_process import data_preprocess

predict_future_step = 2
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
    scalered_features = scaler_X.transform(continuous_features)
    
    # Scale the target (Unemployment Rate)
    scaler_Y = RobustScaler()
    label = df['Unemployment Rate'].astype(float).values.reshape(-1, 1)
    scaler_Y.fit(label)
    scalered_label = scaler_Y.transform(label)
    
    # Combine scaled features with date_features
    features = np.hstack((scalered_features, df[date_features].values))
    label = scalered_label
    
    # Split into training and test sets (Shift by predict_future_step)
    X_train = features[:-predict_future_step]
    y_train = label[:-predict_future_step]
    X_test = features[-predict_future_step:]
    y_test = label[-predict_future_step:]
    
    return X_train, X_test, y_train, y_test, scaler_Y, df


# slice_window function
def slice_window(X_train_scaled, y_train_scaled, future_step, window_size):
    X_train = []
    y_train = []

    # For each index starting from window_size, create a sequence of window_size steps.
    # The target is taken 'future_step' steps ahead.
    for i in range(window_size, len(X_train_scaled) - future_step + 1):
        X_train.append(X_train_scaled[i - window_size:i, :])
        y_train.append(y_train_scaled[i + future_step - 1, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_Y, df = preprocess_data()

    # Test LSTM model
    if os.path.exists("models/lstm_model.h5"):
        model = load_model("models/lstm_model.h5")
        window_size = 12
        future_step = 2
        X_train_lstm, y_train_lstm = slice_window(X_train, y_train, future_step, window_size)
        forecast_lstm = model.predict(X_train_lstm[-future_step:])
        y_pred_future_lstm = scaler_Y.inverse_transform(forecast_lstm)[:, 0]
        y_test_actual = scaler_Y.inverse_transform(y_test)[:, 0]

        # Evaluate LSTM
        mae = mean_absolute_error(y_test_actual, y_pred_future_lstm)
        mse = mean_squared_error(y_test_actual, y_pred_future_lstm)
        print("\nLSTM Model Evaluation:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error:  {mse:.4f}")

    # Test Linear Regression model
    if os.path.exists("models/linear_model.npy"):
        coefficients = np.load("models/linear_model.npy")
        intercept = coefficients[0]
        forecast_linear = X_test.dot(coefficients.T)
        y_pred_future_linear = scaler_Y.inverse_transform(forecast_linear)[:, 0]

        # Evaluate Linear Regression
        mae = mean_absolute_error(y_test_actual, y_pred_future_linear)
        mse = mean_squared_error(y_test_actual, y_pred_future_linear)
        print("\nLinear Regression Model Evaluation:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error:  {mse:.4f}")

    # Plot predictions
    future_dates = df['REF_DATE'].iloc[-future_step:].values
    plt.figure(figsize=(10, 6))
    if os.path.exists("models/lstm_model.h5"):
        plt.plot(future_dates, y_test_actual, marker='o', label='Actual')
        plt.plot(future_dates, y_pred_future_lstm, marker='x', label='LSTM Predicted')
    if os.path.exists("models/linear_model.npy"):
        plt.plot(future_dates, y_pred_future_linear, marker='s', label='Linear Regression Predicted')
    plt.title('Actual vs Predicted Unemployment Rate')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to the output folder
    plot_path = os.path.join("image_folder", "predictions_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
