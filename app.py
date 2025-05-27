from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "chrome-extension://pnfhoobelgilgafgacdnmebgohgknkdg"}})

# Загружаем модель и скейлер для каждой криптовалюты
models = {
    'ETH': load_model('ETH_lstm_model.h5'),
    'BTC': load_model('BTC_lstm_model.h5'),
    'SOL': load_model('SOL_lstm_model.h5'),
    'DOGE': load_model('DOGE_lstm_model.h5')
}
scalers = {
    'ETH': joblib.load('ETH_scaler.pkl'),
    'BTC': joblib.load('BTC_scaler.pkl'),
    'SOL': joblib.load('SOL_scaler.pkl'),
    'DOGE': joblib.load('DOGE_scaler.pkl')
}

@app.route('/predict', methods=['GET'])
def predict():
    crypto = request.args.get('crypto')
    price_str = request.args.get('price')

    if not crypto or not price_str:
        return jsonify({'error': 'Missing crypto or price parameter'}), 400

    try:
        current_price = float(price_str)
        if crypto not in models:
            return jsonify({'error': 'Unsupported cryptocurrency'}), 400

        # Запрашиваем данные за последние 10 дней (часовой интервал для соответствия обучению)
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={crypto}&tsym=USD&limit=240"  # 10 дней * 24 часа
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
            return jsonify({'error': 'Failed to fetch historical data'}), 500

        # Извлекаем цены за последние 240 часов
        prices = [entry['close'] for entry in data['Data']['Data']]

        # Заменяем последнюю цену на текущую (переданную от расширения)
        prices[-1] = current_price

        # Подготовка данных для модели (10 шагов)
        prices = np.array(prices).reshape(-1, 1)
        scaled_prices = scalers[crypto].transform(prices)
        X = scaled_prices[-10:]  # Последние 10 шагов
        X = X.reshape(1, 10, 1)  # Формат для LSTM: [samples, timesteps, features]

        # Делаем прогноз
        model = models[crypto]
        predicted_scaled = model.predict(X, verbose=0)
        predicted_price = scalers[crypto].inverse_transform(predicted_scaled)[0][0]

        return jsonify({'prediction': float(predicted_price)})
    except ValueError:
        return jsonify({'error': 'Price must be a valid number'}), 400
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from CryptoCompare: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    return make_response('', 204)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)