import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from data_preproces_piplines.data_process import data_preprocess

predict_future_step = 2


# preprocess data
def preprocess_data():
    df = data_preprocess()
    df = df[df['GEO'] == 'Canada']

    continuous_columns = ['Participation Rate', 'Population', 'CPI', 'Gross domestic product at market prices', 'Gross fixed capital formation', 'Minimum Wage']

    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df['month'] = df['REF_DATE'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    date_features = ['month_sin', 'month_cos']
    
    scaler_X = RobustScaler()
    continuous_features = df[continuous_columns].astype(float)
    scaler_X.fit(continuous_features)
    scalered_features = scaler_X.transform(continuous_features)
    
    scaler_Y = RobustScaler()
    label = df['Unemployment Rate'].astype(float)
    label = label.values.reshape(-1, 1)
    scaler_Y.fit(label)
    scalered_label = scaler_Y.transform(label)
    
    features = features = np.hstack((scalered_features, df[date_features].values))
    label = scalered_label
    
    X_train = features[:-predict_future_step]
    y_train = label[:-predict_future_step]
    X_test = features[-predict_future_step:]
    y_test = label[-predict_future_step:]
    
    return X_train, X_test, y_train, y_test, scaler_Y
    
def slice_window(X_train_scaled, y_train_scaled, future_step, window_size):
    X_train = []
    y_train = []

    for i in range(window_size, len(X_train_scaled) - future_step +1):
        X_train.append(X_train_scaled[i - window_size:i, :])
        y_train.append(y_train_scaled[i + future_step - 1, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    
    return X_train, y_train

# define model
class LSTM_model():
    def __init__(self, X_train, y_train, epoch, batch_size, validation_split):
        self.X_train = X_train
        self.y_train = y_train
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = Sequential()
        self.LSTM = layers.Bidirectional(LSTM(64, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        self.LSTM2 = LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout_layer = Dropout(0.3)
        self.FC_layer = Dense(1, activation='linear')
        self.training_loss = []
        self.validation_loss = []
        
    def build(self):
        self.model.add(self.LSTM)
        self.model.add(self.dropout_layer)
        self.model.add(self.LSTM2)
        self.model.add(self.dropout_layer)
        self.model.add(self.FC_layer)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())
        
    def train(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, validation_split=self.validation_split, verbose=1, callbacks=[early_stopping])
        self.training_loss = history.history.get('loss', [])
        self.validation_loss = history.history.get('val_loss', [])
        
    def predict(self, X_train):
        return self.model.predict(X_train)
    
    def loss(self):
        return self.training_loss, self.validation_loss
    
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
        
def main():
    # preprocess data
    X_train, X_test, y_train, y_test, scaler_Y = preprocess_data()
    
    # process data through slice window method
    future_step = predict_future_step
    window_size = 12
    X_train, y_train = slice_window(X_train, y_train, future_step, window_size)
    
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    
    # define parameters
    epoch = 100
    batch_size = 8
    validation_split = 0.2
    
    # load & build model
    model = LSTM_model(X_train, y_train, epoch, batch_size, validation_split)
    model.build()
    # train model
    model.train()
    # predict
    forecast = model.predict(X_train[-future_step:])
    print(forecast)
    y_pred_future = scaler_Y.inverse_transform(forecast)[:,0]
    y_test = scaler_Y.inverse_transform(y_test)[:,0]
    print(y_pred_future)
    print(y_test)
    mae = mean_absolute_error(y_test, y_pred_future)
    print(f"Mean Absolute Error: {mae}")
    mse = mean_squared_error(y_test, y_pred_future)
    print(f"Mean Squared Error: {mse}")
    
    loss_plot(model)
    
main()