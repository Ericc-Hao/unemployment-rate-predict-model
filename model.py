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

#from data_processing_piplines.data_process import data_process


# preprocess data
def preprocess_data():
    df = pd.read_csv("./datasets/filtered_data/final_merged_data.csv")
    #df = data_process()
    df = df[df['GEO'] == 'Canada']
    
    #categorical_columns = ['GEO']
    continuous_columns = ['Participation Rate', 'Population', 'CPI', 'Gross domestic product at market prices', 'Gross fixed capital formation']
    #encoded_features = pd.get_dummies(df[categorical_columns], drop_first=False)
    #province_mapping = dict(enumerate(df[categorical_columns[0]].unique()))
    
    date_value = df['REF_DATE']
    date_value = date_value.values.reshape(-1, 1)
    
    scaler_X = RobustScaler()
    continuous_features = df[continuous_columns].astype(float)
    scaler_X.fit(continuous_features)
    scalered_features = scaler_X.transform(continuous_features)
    
    scaler_Y = RobustScaler()
    label = df['Unemployment Rate'].astype(float)
    label = label.values.reshape(-1, 1)
    scaler_Y.fit(label)
    scalered_label = scaler_Y.transform(label)
    
    #features = np.hstack((scalered_features, encoded_features.values))
    features = scalered_features
    label = scalered_label
    
    X_train = features[:-4]
    y_train = label[:-4]
    X_test = features[-4:]
    y_test = label[-4:]
    
    return X_train, X_test, y_train, y_test, scaler_Y
    
def slice_window(X_train_scaled, y_train_scaled, future_step, window_size):
    X_train = []
    y_train = []

    future_steps = future_step
    window_size = window_size

    for i in range(window_size, len(X_train_scaled) - future_steps +1):
        X_train.append(X_train_scaled[i - window_size:i, :])
        y_train.append(y_train_scaled[i + future_steps - 1, 0])

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
        self.LSTM = LSTM(25, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=False)
        self.dropout_layer = Dropout(0.35)
        self.FC_layer = Dense(1, activation='relu')
        self.training_loss = []
        self.validation_loss = []
        
    def build(self):
        self.model.add(self.LSTM)
        self.model.add(self.dropout_layer)
        self.model.add(self.FC_layer)
        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        
    def train(self):
        history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, validation_split=self.validation_split, verbose=1)
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
    future_step = 4
    window_size = 16
    X_train, y_train = slice_window(X_train, y_train, future_step, window_size)
    
    # define parameters
    epoch = 30
    batch_size = 16
    validation_split = 0.2
    
    # load & build model
    model = LSTM_model(X_train, y_train, epoch, batch_size, validation_split)
    model.build()
    # train model
    model.train()
    # predict
    forecast = model.predict(X_train[-future_step:])
    print(X_train[-future_step:])
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