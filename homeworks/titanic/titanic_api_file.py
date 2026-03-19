from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from titanic_model_file import TitanicModel

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class Predict(Resource):
    def post(self):
        passenger = request.get_json()

        model = TitanicModel.get_instance()
        result = model.predict(passenger)

        return jsonify(result)

api.add_resource(Predict, '/predict')