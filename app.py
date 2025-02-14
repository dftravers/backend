import os
import requests
import pandas as pd
import json
import re
import logging
from scipy.stats import poisson
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Football Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ✅ Added error handling for missing data
    if not data or "team1" not in data or "team2" not in data:
        return jsonify({"error": "Both team1 and team2 are required"}), 400

    home_team = data["team1"]
    away_team = data["team2"]

    if home_team == away_team:
        return jsonify({"error": "Teams must be different"}), 400

    return jsonify(full_match_prediction(home_team, away_team))

def full_match_prediction(home_team_name, away_team_name):
    """Wrapper function that fetches data and returns all predictions."""
    df = fetch_understat_xg_data()

    goals = predict_goals(home_team_name, away_team_name, df)
    superbru = best_superbru_prediction(home_team_name, away_team_name, df)
    likely_score = most_likely_score(home_team_name, away_team_name, df)

    logging.debug(f"Full Prediction Output: {goals}, {superbru}, {likely_score}")

    return {
        "Home Team": goals["Home Team"],
        "Away Team": goals["Away Team"],
        "Predicted Goals (Home)": round(goals["Predicted Goals (Home)"], 2),
        "Predicted Goals (Away)": round(goals["Predicted Goals (Away)"], 2),
        "Most Likely Score": likely_score["Most Likely Score"],
        "Best Guess Score": superbru["Best Guess Score"],
        "Expected Points": superbru["Expected Points"]
    }

# ✅ Ensure correct port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
