import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import datetime

def add_lagged_features(df, cols, max_lag=3):
    """Add specified number of lag features for each column in cols."""
    for col in cols:
        for lag in range(1, max_lag+1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def add_difference_features(df, cols):
    """Add difference features (df[col].diff())."""
    for col in cols:
        df[f'{col}_diff'] = df[col].diff()
    return df

def add_month_feature(df, date_col='REF_DATE'):
    """Add a numerical month feature."""
    df['month'] = df[date_col].dt.month
    return df

def create_sequences_linear(X, y, dates, window_size, future_step):
    """
    For a linear model, we'll flatten the time window into one feature vector.
    X shape: (samples, features)
    For each sequence, we take window_size rows and flatten them:
    final shape: (samples, window_size * features)
    """
    X_seq, y_seq, date_seq = [], [], []
    for i in range(window_size, len(X)-future_step+1):
        # Extract the window
        window_X = X[i-window_size:i, :]
        # Flatten the window
        window_flat = window_X.flatten()
        X_seq.append(window_flat)
        y_seq.append(y[i+future_step-1, 0])
        date_seq.append(dates[i+future_step-1])
    return np.array(X_seq), np.array(y_seq), np.array(date_seq)

def preprocess_data(filename="./datasets/filtered_data/final_merged_data.csv", 
                    window_size=4, future_step=1, max_lag=3):
    """Load and preprocess the data, create features, and split into train/test."""
    df = pd.read_csv(filename)
    df = df[df['GEO'] == 'Canada'].copy()
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

    df.sort_values('REF_DATE', inplace=True)
    df.reset_index(drop=True, inplace=True)

    continuous_columns = [
        'Participation Rate', 
        'Population', 
        'CPI', 
        'Gross domestic product at market prices', 
        'Gross fixed capital formation', 
        'Minimum Wage'
    ]

    target_col = 'Unemployment Rate'

    df = add_month_feature(df)
    df = add_difference_features(df, ['CPI', 'Gross domestic product at market prices', 'Gross fixed capital formation'])
    df = add_lagged_features(df, [target_col], max_lag=max_lag)

    # Drop rows with NaN due to shifts/differences
    df.dropna(inplace=True)

    features_to_use = continuous_columns + [
        'month', 
        'CPI_diff', 
        'Gross domestic product at market prices_diff', 
        'Gross fixed capital formation_diff'
    ] + [f'{target_col}_lag{i}' for i in range(1, max_lag+1)]

    # Define train/test split
    test_start_date = pd.to_datetime("2024-06-01")
    test_end_date = pd.to_datetime("2024-11-01")

    train_df = df[df['REF_DATE'] < test_start_date]
    test_df = df[(df['REF_DATE'] >= test_start_date) & (df['REF_DATE'] <= test_end_date)]

    print(f"\nTrain data: {train_df['REF_DATE'].min()} to {train_df['REF_DATE'].max()}")
    print(f"Test data: {test_df['REF_DATE'].min()} to {test_df['REF_DATE'].max()}")

    # Correlation printout
    corr_df = df[[target_col] + features_to_use].corr()
    print("\nCorrelation matrix of features vs target:")
    print(corr_df[target_col].sort_values(ascending=False))

    X_train_all = train_df[features_to_use].astype(float).values
    y_train_all = train_df[[target_col]].astype(float).values
    dates_train_all = train_df['REF_DATE'].values

    X_test_all = test_df[features_to_use].astype(float).values
    y_test_all = test_df[[target_col]].astype(float).values
    dates_test_all = test_df['REF_DATE'].values

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_all)
    y_train_scaled = scaler_Y.fit_transform(y_train_all)

    X_test_scaled = scaler_X.transform(X_test_all)
    y_test_scaled = scaler_Y.transform(y_test_all)

    # Create sequences for linear regression (flattened)
    X_train, y_train, dates_train = create_sequences_linear(X_train_scaled, y_train_scaled, dates_train_all, window_size, future_step)
    X_test, y_test, dates_test = create_sequences_linear(X_test_scaled, y_test_scaled, dates_test_all, window_size, future_step)

    print("\nTraining set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, dates_test, scaler_X, scaler_Y, df, features_to_use, target_col, window_size

def build_and_train_linear_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_dataset(model, X, y, scaler_Y, dataset_name=""):
    """Evaluate model on a given dataset (train or test) and return metrics."""
    y_pred_scaled = model.predict(X)
    y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler_Y.inverse_transform(y.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nEvaluation on {dataset_name} Set:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return mae, mse, rmse, mape

def evaluate_model(model, X_test, y_test, dates_test, scaler_Y):
    """Evaluate the model on the test set and produce evaluation metrics and plots."""
    if len(X_test) == 0:
        print("No test samples available. Adjust parameters or date ranges to produce a test set.")
        return None, None, None, None

    y_pred_scaled = model.predict(X_test)
    # y_test and y_pred_scaled are scaled, inverse transform them
    y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler_Y.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\nEvaluation on Test Set:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    print("\nActual vs. Predicted with Dates:")
    for i in range(len(y_true)):
        date_str = pd.to_datetime(dates_test[i]).strftime('%Y-%m')
        print(f"Date: {date_str} | Actual={y_true[i][0]:.2f}, Predicted={y_pred[i][0]:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10,5))
    plt.plot(dates_test, y_true, label='Actual', marker='o')
    plt.plot(dates_test, y_pred, label='Predicted', marker='x')
    plt.title('Linear Regression Forecast vs. Actual (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Residual analysis
    residuals = y_true.flatten() - y_pred.flatten()
    plt.figure(figsize=(10,5))
    plt.plot(dates_test, residuals, marker='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Over Time (Linear Model)')
    plt.xlabel('Date')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return mae, mse, rmse, mape

def get_future_prediction_linear(model, scaler_X, scaler_Y, df, features_to_use, target_col, window_size=4, last_known_date=pd.to_datetime("2024-11-01")):
    """Predict the next time step beyond last_known_date using linear regression."""
    df_future = df[df['REF_DATE'] <= last_known_date].copy()
    df_future.sort_values('REF_DATE', inplace=True)

    # Extract the last window_size rows
    last_window = df_future[features_to_use].astype(float).values[-window_size:]
    # Scale them
    last_window_scaled = scaler_X.transform(last_window)

    # Flatten
    last_window_flat = last_window_scaled.flatten().reshape(1, -1)

    future_pred_scaled = model.predict(last_window_flat)
    future_pred = scaler_Y.inverse_transform(future_pred_scaled.reshape(-1,1))
    return future_pred[0][0]

def main():
    # Parameters
    window_size = 4
    future_step = 1
    max_lag = 3
    filename = "./datasets/filtered_data/final_merged_data.csv"

    X_train, X_test, y_train, y_test, dates_test, scaler_X, scaler_Y, df_full, features_to_use, target_col, window_size = preprocess_data(
        filename=filename,
        window_size=window_size,
        future_step=future_step,
        max_lag=max_lag
    )

    if X_train.shape[0] == 0:
        print("No training samples formed. Check your date ranges or parameters.")
        return

    model = build_and_train_linear_model(X_train, y_train)

    # Evaluate on training set to get a "loss" measure
    train_mae, train_mse, train_rmse, train_mape = evaluate_dataset(model, X_train, y_train, scaler_Y, dataset_name="Training")

    # Evaluate on test set
    test_mae, test_mse, test_rmse, test_mape = evaluate_model(model, X_test, y_test, dates_test, scaler_Y)
    
    if test_rmse is not None:
        # Compare Training vs Test RMSE in a bar chart
        plt.figure(figsize=(5,5))
        plt.bar(["Train RMSE", "Test RMSE"], [train_rmse, test_rmse], color=['blue', 'orange'])
        plt.ylabel("RMSE")
        plt.title("Training vs Test RMSE Comparison")
        plt.show()

    # Forecast one month beyond the last known date in test set (assume last test date is 2024-11)
    last_known_date = pd.to_datetime("2024-11-01")
    future_pred = get_future_prediction_linear(model, scaler_X, scaler_Y, df_full, features_to_use, target_col, window_size=window_size, last_known_date=last_known_date)
    print(f"\nForecast for 2024-12 (Linear Regression): Predicted Unemployment Rate = {future_pred:.2f}")

if __name__ == "__main__":
    main()
