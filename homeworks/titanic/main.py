from flask import Flask, Blueprint, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# -------- Titanic Model --------
class TitanicModel:
    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        self.titanic_data = sns.load_dataset('titanic')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        df = self.titanic_data
        df.drop(['alive','who','adult_male','class','embark_town','deck'], axis=1, inplace=True)
        df['sex'] = df['sex'].apply(lambda x: 1 if x=='male' else 0)
        df['alone'] = df['alone'].apply(lambda x: 1 if x==True else 0)
        df.dropna(subset=['embarked'], inplace=True)
        onehot = self.encoder.fit_transform(df[['embarked']]).toarray()
        cols = ['embarked_'+str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        df = pd.concat([df, onehot_df], axis=1)
        df.drop(['embarked'], axis=1, inplace=True)
        self.features.extend(cols)
        df.dropna(inplace=True)
        self.titanic_data = df

    def _train(self):
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, passenger):
        df = pd.DataFrame(passenger, index=[0])
        df['sex'] = df['sex'].apply(lambda x: 1 if x=='male' else 0)
        df['alone'] = df['alone'].apply(lambda x: 1 if int(x) else 0)
        onehot = self.encoder.transform(df[['embarked']]).toarray()
        cols = ['embarked_'+str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        df = pd.concat([df, onehot_df], axis=1)
        df.drop(['embarked','name'], axis=1, inplace=True)
        die, survive = np.squeeze(self.model.predict_proba(df))
        return {'survival_probability': round(float(survive),4),
                'death_probability': round(float(die),4)}

# -------- Flask API --------
app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class Predict(Resource):
    def post(self):
        passenger = request.get_json()
        model = TitanicModel.get_instance()
        result = model.predict(passenger)
        return result

api.add_resource(Predict, '/predict')
app.register_blueprint(titanic_api)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)