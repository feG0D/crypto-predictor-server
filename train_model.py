import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import joblib

# Функция для загрузки исторических данных
def fetch_historical_data(crypto, days=180):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto}&tsym=USD&limit={days}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for {crypto}: {response.status_code}")
    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    return df['close'].values

# Подготовка данных для LSTM
def prepare_data(data, time_steps=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Создание и обучение модели
def train_lstm_model(crypto):
    try:
        # Загружаем данные за последние 180 дней
        data = fetch_historical_data(crypto, days=180)
        if len(data) < 10:
            raise ValueError(f"Insufficient data for {crypto}")

        # Подготовка данных
        time_steps = 10
        X, y, scaler = prepare_data(data, time_steps)

        # Разделяем данные на обучающую и тестовую выборки
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Создаём модель LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        # Обучаем модель
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Сохраняем модель и scaler
        model.save(f"{crypto}_lstm_model.h5")
        joblib.dump(scaler, f"{crypto}_scaler.pkl")
        print(f"Model and scaler for {crypto} saved successfully!")
    except Exception as e:
        print(f"Error training model for {crypto}: {str(e)}")

# Обучаем модели для всех криптовалют
if __name__ == "__main__":
    cryptos = ['BTC', 'ETH', 'SOL', 'DOGE']
    for crypto in cryptos:
        print(f"Training model for {crypto}...")
        train_lstm_model(crypto)