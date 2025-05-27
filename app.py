from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sqlite3
import os

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

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (user_id TEXT PRIMARY KEY, chat_id TEXT)''')
    conn.commit()
    conn.close()

# Эндпоинт для предсказаний
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

        # Отправка уведомления, если разница больше 5%
        price_diff = abs(predicted_price - current_price) / current_price * 100
        if price_diff > 5:
            user_id = request.args.get('userId')  # Получаем userId из запроса (добавлен в script.js)
            if user_id:
                notification_message = f"Предупреждение по цене {crypto}: Цена может измениться на {price_diff:.2f}% за 1 день! Текущая: ${current_price}, Прогноз: ${predicted_price:.2f}"
                send_telegram_notification(user_id, notification_message, 'ru')  # Язык по умолчанию, можно улучшить

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

# Эндпоинт для отправки уведомлений
def send_telegram_notification(user_id, message, lang):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT chat_id FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()

    if result:
        chat_id = result[0]
        TELEGRAM_BOT_TOKEN = '8176459174:AAHYP9fGzmGbnoUnmTplk7OxUGyeEuTqA5U'
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(url, data=payload)
            data = response.json()
            if not data.get('ok'):
                print(f'Ошибка отправки в Telegram: {data.get("description")}')
        except Exception as e:
            print(f'Ошибка при отправке уведомления: {str(e)}')
    else:
        print(f'Пользователь {user_id} не подписан на уведомления')
    conn.close()

# Эндпоинт для обработки подписки от бота
@app.route('/subscribe_telegram', methods=['POST'])
def subscribe_telegram():
    user_id = request.args.get('userId')
    chat_id = request.args.get('chatId')

    if not user_id or not chat_id:
        return jsonify({'ok': False, 'message': 'Missing userId or chatId'}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (user_id, chat_id) VALUES (?, ?)", (user_id, chat_id))
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'message': 'Subscribed successfully'})

if __name__ == '__main__':
    init_db()  # Инициализируем базу данных при запуске
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)