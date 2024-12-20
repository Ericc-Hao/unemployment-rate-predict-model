import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import optimizers, regularizers

from models.data_preproces_piplines.data_process import data_preprocess
from models.data_preproces_piplines.adjust_data_process import adjust_data_process

predict_future_step = 2

def adjust_preprocess_data():
    # Load and preprocess the data
    df = adjust_data_process()

    # Convert REF_DATE to datetime and extract month
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df['month'] = df['REF_DATE'].dt.month

    # Create cyclical monthly features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    date_features = ['month_sin', 'month_cos']
    
    # Scale continuous features using RobustScaler
    scaler_X = RobustScaler()
    # scaler_X = StandardScaler()

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    continuous_features = df[numeric_columns].astype(float)
    scaler_X.fit(continuous_features)
    scaled_features = scaler_X.transform(continuous_features)
    
    # Scale the target (Unemployment Rate)
    scaler_Y = RobustScaler()
    # scaler_Y = StandardScaler()
    label = df['Canada-Unemployment rate'].astype(float).values.reshape(-1, 1)
    scaler_Y.fit(label)
    scaled_label = scaler_Y.transform(label)
    
    # Combine scaled features with date_features
    features = np.hstack((scaled_features, df[date_features].values))
    label = scaled_label
    
    # Split into training and test sets (Shift by predict_future_step)
    X_train = features[:-predict_future_step]
    y_train = label[:-predict_future_step]
    X_test = features[-predict_future_step:]
    y_test = label[-predict_future_step:]
    
    return X_train, X_test, y_train, y_test, scaler_Y, df


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
    # scaler_X = RobustScaler()
    scaler_X = StandardScaler()
    continuous_features = df[continuous_columns].astype(float)
    # continuous_features = df.astype(float)
    scaler_X.fit(continuous_features)
    scalered_features = scaler_X.transform(continuous_features)
    
    # Scale the target (Unemployment Rate)
    # scaler_Y = RobustScaler()
    scaler_Y = StandardScaler()
    label = df['Unemployment Rate'].astype(float).values.reshape(-1, 1)
    # label = df['Canada-Unemployment rate'].astype(float).values.reshape(-1, 1)
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


class LSTM_model:
    def __init__(self, X_train, y_train, epoch, batch_size, validation_split):
        self.X_train = X_train
        self.y_train = y_train
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.model = None
        self.training_loss = []
        self.validation_loss = []

    def build(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='linear'))
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=MeanSquaredError()
        )
        print("LSTM model built successfully.")

    def train(self):        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epoch,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=1,
        )
        
        # Store training history
        self.training_loss = history.history.get('loss', [])
        self.validation_loss = history.history.get('val_loss', [])
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        self.model.save("models/lstm_model.h5")
        with open('models/lstm_model.pkl', 'wb') as f:
            pickle.dump('models/lstm_model.h5', f)
        print("LSTM model trained and saved as lstm_model.pkl")

        np.save("models/training_loss.npy", self.training_loss)
        np.save("models/validation_loss.npy", self.validation_loss)
        print("Loss history saved as training_loss.npy and validation_loss.npy")

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def save(self, filename="models/linear_model.pkl"):
        os.makedirs("models", exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Linear Regression model coefficients saved to {filename}")

if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_Y, df = preprocess_data()
    # X_train, X_test, y_train, y_test, scaler_Y, df = adjust_preprocess_data()


    # Train and save LSTM model
    print(f"\n{'Training LSTM model'.center(50, '-')}")
    window_size = 12
    future_step = 2
    X_train_lstm, y_train_lstm = slice_window(X_train, y_train, future_step, window_size)

    lstm_model = LSTM_model(X_train_lstm, y_train_lstm, epoch=100, batch_size=8, validation_split=0.2)
    lstm_model.build()
    lstm_model.train()
    print(f"{'Finished Train LSTM model'.center(50, '-')}")


    print(f"\n{'Training Linear Regression model'.center(50, '-')}")
    # Train and save Linear Regression model
    linear_model = LinearRegressionModel()
    linear_model.train(X_train, y_train)
    linear_model.save()
    print(f"{'Finished Train Linear Regression model'.center(50, '-')}")