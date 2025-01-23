from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app)

# Enable CORS for all routes
CORS(app)

# Load the model
model = joblib.load('D:/water_potability/WaterPotability/models/SVC.joblib')

# Swagger configuration
SWAGGER_URL = '/docs'  # URL for accessing Swagger UI
API_URL = '/swagger.json'  # API documentation JSON file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Water Potability Prediction API"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route('/swagger.json')
def swagger_json():
    """Swagger API Definition"""
    return jsonify({
        "swagger": "2.0",
        "info": {
            "title": "Water Potability Prediction API",
            "description": "API for predicting water potability based on input features.",
            "version": "1.0.0"
        },
        "host": "localhost:5000",
        "basePath": "/",
        "schemes": ["http"],
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Predict water potability",
                    "description": "Provide features to get the potability prediction.",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "feature1": {"type": "number"},
                                    "feature2": {"type": "number"},
                                    "feature3": {"type": "number"},
                                    "feature4": {"type": "number"},
                                    "feature5": {"type": "number"},
                                    "feature6": {"type": "number"},
                                    "feature7": {"type": "number"},
                                    "feature8": {"type": "number"},
                                    "feature9": {"type": "number"}
                                },
                                "required": ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "prediction": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    })


class Predict(Resource):
    def post(self):
        try:
            data = request.get_json(force=True)
            # Extract features from input
            features = np.array([
                data['feature1'], data['feature2'], data['feature3'],
                data['feature4'], data['feature5'], data['feature6'],
                data['feature7'], data['feature8'], data['feature9']
            ])
            prediction = model.predict([features])
            return jsonify({'prediction': str(prediction[0])})
        except Exception as e:
            return {"error": str(e)}, 400


# Add the endpoint to the API
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
