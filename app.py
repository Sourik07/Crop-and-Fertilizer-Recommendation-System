from flask import Flask, jsonify,request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the ML model
joblib_file = "fert-model.pkl"
dtmodel = joblib.load(joblib_file)

joblib_crop = "kn_classifier_model.pkl"
kn_model = joblib.load(joblib_crop)

@app.route('/fertilizer/', methods=['GET'])
def predict():
    try:
        nitrogen = int(request.args.get('nitrogen'))
        phosphorous = int(request.args.get('phosphorous'))
        potassium = int(request.args.get('potassium'))
        temperature = int(request.args.get('temperature'))
        humidity = int(request.args.get('humidity'))
        moisture = int(request.args.get('moisture'))
        # Predict using the ML model
        prediction = dtmodel.predict([[temperature, humidity, moisture, nitrogen, potassium, phosphorous]])


        # Return prediction as JSON response
        response = {'prediction': prediction[0]}
        return jsonify(response), 200
    except Exception as e:
        # Return error message if there's any exception
        response = {'error': str(e)}
        return jsonify(response), 400


@app.route('/crop/', methods=['GET'])
def predict2():
    try:
        nitrogen = int(request.args.get('nitrogen'))
        phosphorous = int(request.args.get('phosphorous'))
        potassium = int(request.args.get('potassium'))
        temperature = int(request.args.get('temperature'))
        humidity = int(request.args.get('humidity'))
        ph_level = int(request.args.get('ph_level'))
        rainfall = int(request.args.get('rainfall'))
        # Predict using the ML model
        prediction2 = kn_model.predict([[nitrogen, phosphorous, potassium, temperature, humidity, ph_level, rainfall]])

        # Return prediction as JSON response
        response = {'prediction': prediction2[0]}
        return jsonify(response), 200

    except Exception as e:
        # Return error message if there's any exception
        response = {'error': str(e)}
        return jsonify(response), 400
        


if __name__ == '__main__':
    app.run()
