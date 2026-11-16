import pickle

from flask import Flask, request, jsonify

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('cardio')


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    # bmi считаем на сервере, чтобы клиенту не нужно было его передавать
    patient['bmi'] = patient['weight'] / (patient['height'] / 100) ** 2

    X = dv.transform([patient])
    probability = float(model.predict_proba(X)[0, 1])

    return jsonify({
        'cardio_probability': round(probability, 3),
        'cardio_risk': probability >= 0.5,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)