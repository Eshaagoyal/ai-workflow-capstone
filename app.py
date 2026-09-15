from flask import Flask, request, jsonify
from model import model_train, model_predict
import os

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(silent=True) or {}
    test_mode = data.get('mode') == 'test'
    data_dir = os.path.join(".", "cs-train")
    
    try:
        models = model_train(data_dir, test=test_mode)
        return jsonify({"status": "success", "trained_models": list(models.keys())}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' payload"}), 400
        
    query = data['query']
    country = query.get('country', 'all')
    year = int(query.get('year', 2018))
    month = int(query.get('month', 11))
    day = int(query.get('day', 20))
    test_mode = data.get('mode') == 'test'
    
    try:
        result = model_predict(country, year, month, day, test=test_mode)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logs/<filename>', methods=['GET'])
def get_logs(filename):
    log_path = os.path.join(".", "logs", filename)
    if not os.path.exists(log_path):
        return jsonify({"error": "File not found"}), 404
    with open(log_path, 'r') as f:
        content = f.read()
    return content, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)