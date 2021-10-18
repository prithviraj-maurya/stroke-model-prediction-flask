from flask import Flask, request, jsonify
from flask_cors import CORS

from diseases.stroke_predicitons import make_stroke_prediction
from diseases.kidney import make_kidney_prediction
from diseases.cirrosis import make_cirrosis_prediction
import risk_recommendation

## Flask api
app = Flask(__name__)
CORS(app)

@app.route('/predict/stroke', methods=['POST'])
def stroke():
    data = request.json
    predictions, feature_importances = make_stroke_prediction(data)
    # Return on a JSON format
    return jsonify(
        prediction= predictions,
        feature_importances=feature_importances.tolist(),
        risks=risk_recommendation.stroke['risks'],
        recommendations=risk_recommendation.stroke['recommendations']
    )

@app.route('/predict/kidney', methods=['POST'])
def kidney():
    data = request.json
    predictions, feature_importances = make_kidney_prediction(data)
    # Return on a JSON format
    return jsonify(
        prediction= predictions,
        feature_importances=feature_importances.tolist(),
        risks=risk_recommendation.kidney['risks'],
        recommendations=risk_recommendation.kidney['recommendations']
    )

@app.route('/predict/cirrosis', methods=['POST'])
def cirrosis():
    data = request.json
    predictions, feature_importances = make_cirrosis_prediction(data)
    # Return on a JSON format
    return jsonify(
        prediction= predictions,
        feature_importances=feature_importances,
        risks=risk_recommendation.kidney['risks'],
        recommendations=risk_recommendation.kidney['recommendations']
    )

if __name__ == "__main__":
    app.run()