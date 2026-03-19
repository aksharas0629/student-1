from flask import Flask, render_template, request
import requests

app = Flask(__name__)
API_URL = "http://127.0.0.1:5003/api/titanic/predict"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', form_values={}, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    passenger = {
        'pclass': int(form_data.get('pclass', 3)),
        'sex': 'male' if form_data.get('sex','1')=='1' else 'female',
        'age': float(form_data.get('age',22)),
        'sibsp': int(form_data.get('sibsp',0)),
        'parch': int(form_data.get('parch',0)),
        'fare': float(form_data.get('fare',7.25)),
        'alone': int(form_data.get('alone',0)),
        'embarked': 'C' if form_data.get('embarked_C','0')=='1' else ('Q' if form_data.get('embarked_Q','0')=='1' else 'S'),
        'name': 'User Passenger'
    }

    try:
        response = requests.post(API_URL, json=passenger)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {'error': str(e)}

    return render_template('index.html', form_values=form_data, result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)