from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import json
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
    url = "https://understat.com/league/EPL"
    response = requests.get(url)
    
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch data"}), 500

    raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
    if not raw_data:
        return jsonify({"error": "Could not locate teamsData"}), 500

    json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

    team_stats = [{"Team": team_info["title"]} for team_id, team_info in json_data.items()]
    
    return pd.DataFrame(team_stats)

@app.route('/')
def home():
    """Root route to check if the server is running."""
    return jsonify({"message": "Backend is running!"})

@app.route('/teams', methods=['GET'])
def get_teams():
    """Endpoint to return a list of teams."""
    df = fetch_understat_xg_data()
    return jsonify(df['Team'].tolist())

@app.route('/predict', methods=['POST'])  # Removed the trailing slash issue
def predict():
    """Predict match score."""
    data = request.json
    home_team = data.get('team1')  # Ensure this matches frontend's request
    away_team = data.get('team2')

    if not home_team or not away_team:
        return jsonify({"error": "Both teams must be selected"}), 400

    predicted_score = f"{home_team} 2 - 1 {away_team}"  # Dummy score, replace with actual model

    return jsonify({'prediction': predicted_score})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Get PORT from environment
    app.run(debug=True, host="0.0.0.0", port=port)  # Bind to 0.0.0.0
