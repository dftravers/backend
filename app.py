import os
import requests
import pandas as pd
import json
import re
import logging
from scipy.stats import poisson
from flask import Flask, request, jsonify
from flask_cors import CORS

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Football Prediction API!"

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
    try:
        url = "https://understat.com/league/EPL"
        response = requests.get(url)
        response.raise_for_status()

        raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
        if not raw_data:
            raise ValueError("Could not locate the teamsData variable in the page.")
        
        json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

        return json_data  # Returning raw data as it was originally

    except Exception as e:
        logging.error(f"Error fetching xG data: {str(e)}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "team1" not in data or "team2" not in data:
        return jsonify({"error": "Both 'team1' and 'team2' are required"}), 400

    home_team = data["team1"]
    away_team = data["team2"]

    if home_team == away_team:
        return jsonify({"error": "Teams must be different"}), 400

    response_data = fetch_understat_xg_data()
    return jsonify(response_data)  # This was the original way data was returned

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
