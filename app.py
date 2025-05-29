from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sqlite3
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "chrome-extension://pnfhoobelgilgafgacdnmebgohgknkdg"},
                    r"/get_chat_id": {"origins": "*"},
                    r"/subscribe_telegram": {"origins": "*"}})

# URL сервиса bot.py для отправки уведомлений
TELEGRAM_NOTIFICATION_URL = "https://telegram-bot-x0i8.onrender.com/send_telegram_notification"

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
    period = request.args.get('period')  # Новый параметр: '1m' или '24h'
    user_id = request.args.get('userId')  # Получаем userId из запроса

    if not crypto or not price_str or not period:
        return jsonify({'error': 'Missing crypto, price, or period parameter'}), 400

    if period not in ['1m', '24h']:
        return jsonify({'error': 'Invalid period parameter. Use "1m" or "24h"'}), 400

    try:
        current_price = float(price_str)
        if crypto not in models:
            return jsonify({'error': 'Unsupported cryptocurrency'}), 400

        # Определяем интервал данных в зависимости от периода
        if period == '1m':
            # Для прогноза на 1 минуту используем данные за последние 120 минут (минутный интервал)
            url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={crypto}&tsym=USD&limit=120"
            time_steps = 10  # Используем последние 10 минут для прогноза
        else:  # period == '24h'
            # Для прогноза на 24 часа используем данные за последние 20 дней (часовой интервал)
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={crypto}&tsym=USD&limit=480"  # 20 дней * 24 часа
            time_steps = 10  # Используем последние 10 часов для прогноза

        # Запрашиваем исторические данные
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Получены данные для {crypto}: {len(data['Data']['Data'])} записей")

        if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:
            return jsonify({'error': 'Failed to fetch historical data'}), 500

        # Извлекаем цены
        prices = [entry['close'] for entry in data['Data']['Data']]
        print(f"Цены за последние шаги: {prices[-10:]}")  # Выводим последние 10 цен

        # Заменяем последнюю цену на текущую (переданную от расширения)
        prices[-1] = current_price
        print(f"Текущая цена {crypto}: {current_price}")

        # Подготовка данных для модели
        prices = np.array(prices).reshape(-1, 1)
        scaled_prices = scalers[crypto].transform(prices)
        X = scaled_prices[-time_steps:]  # Последние time_steps шагов
        print(f"Масштабированные данные для прогноза: {X.flatten()}")
        X = X.reshape(1, time_steps, 1)  # Формат для LSTM: [samples, timesteps, features]

        # Делаем прогноз
        model = models[crypto]
        predicted_scaled = model.predict(X, verbose=0)
        predicted_price = scalers[crypto].inverse_transform(predicted_scaled)[0][0]
        print(f"Прогноз (масштабированный): {predicted_scaled}, Прогноз (обратное масштабирование): {predicted_price}")

        # Простая корректировка: если текущая цена выше среднего за последние 10 шагов, добавляем небольшой рост
        last_10_prices = prices[-10:].flatten()
        avg_price = np.mean(last_10_prices)
        if current_price > avg_price:
            predicted_price = current_price + (current_price - avg_price) * 0.1  # Увеличиваем на 10% разницы
            print(f"Корректировка прогноза (рост): {predicted_price}")

        # Отправка уведомления, если разница больше 5%
        price_diff = abs(predicted_price - current_price) / current_price * 100
        if price_diff > 5 and user_id:
            period_text = "1 минуту" if period == '1m' else "24 часа"
            notification_message = f"Предупреждение по цене {crypto}: Цена может измениться на {price_diff:.2f}% за {period_text}! Текущая: ${current_price}, Прогноз: ${predicted_price:.2f}"
            # Отправляем запрос на bot.py для уведомления
            try:
                response = requests.post(
                    TELEGRAM_NOTIFICATION_URL,
                    params={'userId': user_id, 'message': notification_message, 'lang': 'ru'}
                )
                response.raise_for_status()
                print(f"Уведомление отправлено через bot.py: {response.json()}")
            except requests.RequestException as e:
                print(f"Ошибка при отправке уведомления через bot.py: {str(e)}")

        return jsonify({'prediction': float(predicted_price)})
    except ValueError:
        return jsonify({'error': 'Price must be a valid number'}), 400
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from CryptoCompare: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Эндпоинт для получения chat_id
@app.route('/get_chat_id', methods=['GET'])
def get_chat_id():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'ok': False, 'message': 'Missing userId'}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT chat_id FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()

    if result:
        return jsonify({'ok': True, 'chat_id': result[0]})
    return jsonify({'ok': False, 'message': 'User not found'}), 404

# Эндпоинт для обработки подписки от бота
@app.route('/subscribe_telegram', methods=['POST'])
def subscribe_telegram():
    user_id = request.args.get('userId')
    chat_id = request.args.get('chatId')

    if not user_id or not chat_id:
        return jsonify({'ok': false, 'message': 'Missing userId or chatId'}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (user_id, chat_id) VALUES (?, ?)", (user_id, chat_id))
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'message': 'Subscribed successfully'})

@app.route('/favicon.ico')
def favicon():
    return make_response('', 204)

if __name__ == '__main__':
    init_db()  # Инициализируем базу данных при запуске
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)