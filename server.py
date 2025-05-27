from flask import Flask, request, jsonify, make_response

from flask_cors import CORS

import requests

import statistics



app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "chrome-extension://pnfhoobelgilgafgacdnmebgohgknkdg"}})



@app.route('/predict', methods=['GET'])

def predict():

    crypto = request.args.get('crypto')

    price_str = request.args.get('price')

    

    # Проверяем, что параметры переданы

    if not crypto or not price_str:

        return jsonify({'error': 'Missing crypto or price parameter'}), 400

    

    try:

        current_price = float(price_str)

        

        # Запрашиваем минутные данные за последние 10 минут

        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={crypto}&tsym=USD&limit=10"

        response = requests.get(url)

        response.raise_for_status()  # Проверяем, успешен ли запрос

        data = response.json()

        

        # Проверяем, что данные получены

        if 'Data' not in data or 'Data' not in data['Data'] or not data['Data']['Data']:

            return jsonify({'error': 'Failed to fetch historical data'}), 500

        

        # Извлекаем цены закрытия за последние 10 минут

        prices = [entry['close'] for entry in data['Data']['Data']]

        

        # Вычисляем изменения цен между минутами

        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

        

        # Если изменений нет, используем текущую цену

        if not price_changes:

            return jsonify({'prediction': current_price})

        

        # Вычисляем среднее процентное изменение

        avg_change = statistics.mean(price_changes)

        

        # Делаем прогноз на 1 минуту вперёд

        predicted_price = current_price * (1 + avg_change)

        

        return jsonify({'prediction': predicted_price})

    except ValueError:

        return jsonify({'error': 'Price must be a valid number'}), 400

    except requests.RequestException as e:

        return jsonify({'error': f'Failed to fetch data from CryptoCompare: {str(e)}'}), 500

    except Exception as e:

        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500



@app.route('/favicon.ico')

def favicon():

    return make_response('', 204)  # Возвращаем пустой ответ с кодом 204 (No Content)



if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)