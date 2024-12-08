import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Input
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime

from data_preproces_piplines.data_process import data_preprocess

predict_future_step = 2

# preprocess_data function
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


# LSTM_model class
class LSTM_model():
    def __init__(self, X_train, y_train, epoch, batch_size, validation_split):
        self.X_train = X_train
        self.y_train = y_train
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.model = Sequential()
        
        # Define model architecture using Input layer first
        self.model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        # Add LSTM layers without specifying input_shape directly
        self.model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='linear'))
        
        self.training_loss = []
        self.validation_loss = []
        
    def build(self):
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
            loss=tf.keras.losses.MeanSquaredError()
        )
        
    def train(self):
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train the model
        history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=self.epoch, 
            batch_size=self.batch_size, 
            validation_split=self.validation_split, 
            verbose=1, 
            callbacks=[early_stopping]
        )
        
        # Store training history
        self.training_loss = history.history.get('loss', [])
        self.validation_loss = history.history.get('val_loss', [])
        
    def predict(self, X_train):
        # Generate predictions
        return self.model.predict(X_train)
    
    def loss(self):
        return self.training_loss, self.validation_loss


# loss_plot function
def loss_plot(model):
    training_loss, validation_loss = model.loss()
    
    plt.figure(figsize=(15, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Loss for LSTM Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# main function
def main():
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_Y, df = preprocess_data()
    
    # Prepare data with sliding window
    future_step = predict_future_step
    window_size = 12
    X_train, y_train = slice_window(X_train, y_train, future_step, window_size)
    
    # Print dataset shapes
    print("\n" + "-"*50)
    print("Data Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("-"*50 + "\n")
    
    # Model parameters
    epoch = 100
    batch_size = 8
    validation_split = 0.2
    
    # Initialize, build and train the model
    model = LSTM_model(X_train, y_train, epoch, batch_size, validation_split)
    model.build()
    print("Training the model...\n")
    model.train()
    
    # Predict using the last future_step from the training data
    forecast = model.predict(X_train[-future_step:])
    
    # Inverse transform predictions and actual test values
    y_pred_future = scaler_Y.inverse_transform(forecast)[:, 0]
    y_test = scaler_Y.inverse_transform(y_test)[:, 0]
    
    # Retrieve the corresponding future dates from the dataframe
    # We know that y_test corresponds to the last 'future_step' entries of the original data
    future_dates = df['REF_DATE'].iloc[-future_step:].values
    
    # Print predictions and actual values with corresponding dates
    print("\n" + "-"*50)
    print("Future Predictions vs Actual (by Date):")
    for i, date in enumerate(future_dates):
        print(f"Date: {datetime.strptime(str(date)[:10], '%Y-%m-%d').strftime('%B %Y')} | Predicted = {y_pred_future[i]:.2f}, Actual = {y_test[i]:.2f}")
    print("-"*50 + "\n")
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred_future)
    mse = mean_squared_error(y_test, y_pred_future)
    print("Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error:  {mse:.4f}")
    print("-"*50 + "\n")
    
    # Plot the loss curves
    loss_plot(model)


# Run the main function
if __name__ == "__main__":
    main()
