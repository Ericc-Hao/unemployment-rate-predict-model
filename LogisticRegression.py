import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import datetime

def add_lagged_features(df, cols, max_lag=3):
    for col in cols:
        for lag in range(1, max_lag+1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def add_difference_features(df, cols):
    for col in cols:
        df[f'{col}_diff'] = df[col].diff()
    return df

def add_month_feature(df, date_col='REF_DATE'):
    df['month'] = df[date_col].dt.month
    return df

def create_sequences_linear(X, y, dates, window_size, future_step):
    X_seq, y_seq, date_seq = [], [], []
    for i in range(window_size, len(X)-future_step+1):
        window_X = X[i-window_size:i, :]
        window_flat = window_X.flatten()
        X_seq.append(window_flat)
        y_seq.append(y[i+future_step-1])
        date_seq.append(dates[i+future_step-1])
    return np.array(X_seq), np.array(y_seq), np.array(date_seq)

def categorize_unemployment(rate):
    # Example binning: Adjust thresholds as needed
    if rate < 5.0:
        return 0  # Low
    elif rate < 7.0:
        return 1  # Medium
    else:
        return 2  # High

def preprocess_data(filename="./datasets/filtered_data/final_merged_data.csv", 
                    window_size=4, future_step=1, max_lag=3):
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

    df.dropna(inplace=True)

    features_to_use = continuous_columns + [
        'month', 
        'CPI_diff', 
        'Gross domestic product at market prices_diff', 
        'Gross fixed capital formation_diff'
    ] + [f'{target_col}_lag{i}' for i in range(1, max_lag+1)]

    # Convert unemployment rate to categories
    df['Unemployment_Category'] = df[target_col].apply(categorize_unemployment)

    test_start_date = pd.to_datetime("2024-06-01")
    test_end_date = pd.to_datetime("2024-11-01")

    train_df = df[df['REF_DATE'] < test_start_date]
    test_df = df[(df['REF_DATE'] >= test_start_date) & (df['REF_DATE'] <= test_end_date)]

    print(f"\nTrain data: {train_df['REF_DATE'].min()} to {train_df['REF_DATE'].max()}")
    print(f"Test data: {test_df['REF_DATE'].min()} to {test_df['REF_DATE'].max()}")

    X_train_all = train_df[features_to_use].astype(float).values
    y_train_all = train_df['Unemployment_Category'].values
    dates_train_all = train_df['REF_DATE'].values

    X_test_all = test_df[features_to_use].astype(float).values
    y_test_all = test_df['Unemployment_Category'].values
    dates_test_all = test_df['REF_DATE'].values

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_all)
    X_test_scaled = scaler_X.transform(X_test_all)

    X_train, y_train, dates_train = create_sequences_linear(X_train_scaled, y_train_all, dates_train_all, window_size, future_step)
    X_test, y_test, dates_test = create_sequences_linear(X_test_scaled, y_test_all, dates_test_all, window_size, future_step)

    print("\nTraining set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, dates_test, scaler_X, df, features_to_use, target_col, window_size

def build_and_train_multinomial_model(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, dates_test):
    if len(X_test) == 0:
        print("No test samples available.")
        return

    y_pred = model.predict(X_test)

    # Determine which classes are present
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    class_map = {0: 'Low', 1: 'Medium', 2: 'High'}

    # Filter target_names to those actually present
    present_target_names = [class_map[c] for c in unique_classes]

    # Print classification report only for classes present
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred, labels=unique_classes, target_names=present_target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=unique_classes))

    print("\nActual vs. Predicted Categories:")
    for i in range(len(y_test)):
        date_str = pd.to_datetime(dates_test[i]).strftime('%Y-%m')
        actual_cat = class_map[y_test[i]]
        pred_cat = class_map[y_pred[i]]
        print(f"Date: {date_str} | Actual={actual_cat}, Predicted={pred_cat}")

def get_future_prediction_multinomial(model, scaler_X, df, features_to_use, window_size=4, last_known_date=pd.to_datetime("2024-11-01")):
    df_future = df[df['REF_DATE'] <= last_known_date].copy()
    df_future.sort_values('REF_DATE', inplace=True)

    # Extract the last window_size rows
    last_window = df_future[features_to_use].astype(float).values[-window_size:]
    # Scale them
    last_window_scaled = scaler_X.transform(last_window)

    # Flatten
    last_window_flat = last_window_scaled.flatten().reshape(1, -1)

    future_pred = model.predict(last_window_flat)
    return future_pred[0]

def main():
    window_size = 4
    future_step = 1
    max_lag = 3
    filename = "./datasets/filtered_data/final_merged_data.csv"

    X_train, X_test, y_train, y_test, dates_test, scaler_X, df_full, features_to_use, target_col, window_size = preprocess_data(
        filename=filename,
        window_size=window_size,
        future_step=future_step,
        max_lag=max_lag
    )

    if X_train.shape[0] == 0:
        print("No training samples formed. Check your date ranges or parameters.")
        return

    model = build_and_train_multinomial_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, dates_test)

    # Forecast one month beyond the last known date
    last_known_date = pd.to_datetime("2024-11-01")
    future_pred_class = get_future_prediction_multinomial(model, scaler_X, df_full, features_to_use, window_size=window_size, last_known_date=last_known_date)
    class_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    future_cat = class_map[future_pred_class]
    print(f"\nForecast for 2024-12 (Multinomial Classification): Predicted Category = {future_cat}")

if __name__ == "__main__":
    main()
