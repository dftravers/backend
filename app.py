import os
import requests
import pandas as pd
import json
import re
import logging
from scipy.stats import poisson
from flask import Flask, request, jsonify
from flask_cors import CORS

# ✅ Enable logging for debugging
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

        team_stats = []
        for team_id, team_info in json_data.items():
            home_matches = [match for match in team_info['history'] if match['h_a'] == 'h']
            away_matches = [match for match in team_info['history'] if match['h_a'] == 'a']

            team_stats.append({
                'Team': team_info['title'],
                'Home_Games_Played': len(home_matches),
                'xG_home': sum(float(match['xG']) for match in home_matches),
                'xGA_home': sum(float(match['xGA']) for match in home_matches),
                'Away_Games_Played': len(away_matches),
                'xG_away': sum(float(match['xG']) for match in away_matches),
                'xGA_away': sum(float(match['xGA']) for match in away_matches),
            })

        df = pd.DataFrame(team_stats)

        df['Avg_xG_home'] = df['xG_home'] / df['Home_Games_Played'].replace(0, 1)
        df['Avg_xGA_home'] = df['xGA_home'] / df['Home_Games_Played'].replace(0, 1)
        df['Avg_xG_away'] = df['xG_away'] / df['Away_Games_Played'].replace(0, 1)
        df['Avg_xGA_away'] = df['xGA_away'] / df['Away_Games_Played'].replace(0, 1)

        return df

    except Exception as e:
        logging.error(f"Error fetching xG data: {str(e)}")
        return None

def predict_goals(home_team_name, away_team_name, data):
    """Predict the number of goals for each team using xG and xGA."""
    avg_xGA_away_per_game = data['xGA_away'].sum() / data['Away_Games_Played'].sum()
    avg_xGA_home_per_game = data['xGA_home'].sum() / data['Home_Games_Played'].sum()

    home_team = data[data['Team'] == home_team_name].iloc[0]
    away_team = data[data['Team'] == away_team_name].iloc[0]

    home_expected_goals = (
        home_team['Avg_xG_home']
        * away_team['Avg_xGA_away']
        / avg_xGA_away_per_game
    )
    away_expected_goals = (
        away_team['Avg_xG_away']
        * home_team['Avg_xGA_home']
        / avg_xGA_home_per_game
    )

    return {
        'Home Team': home_team_name,
        'Away Team': away_team_name,
        'Predicted Goals (Home)': round(home_expected_goals, 2),
        'Predicted Goals (Away)': round(away_expected_goals, 2)
    }

def most_likely_score(home_team_name, away_team_name, data):
    """Find the most probable scoreline based on Poisson probabilities."""
    result = predict_goals(home_team_name, away_team_name, data)
    home_expected_goals = result['Predicted Goals (Home)']
    away_expected_goals = result['Predicted Goals (Away)']

    max_goals = 6
    max_probability = 0
    best_score = (0, 0)

    for actual_home in range(max_goals + 1):
        for actual_away in range(max_goals + 1):
            prob = poisson.pmf(actual_home, home_expected_goals) * poisson.pmf(actual_away, away_expected_goals)
            if prob > max_probability:
                max_probability = prob
                best_score = (actual_home, actual_away)

    return {
        'Most Likely Score': f"{best_score[0]}-{best_score[1]}"
    }

def best_superbru_prediction(home_team_name, away_team_name, data):
    """Find the best Superbru prediction by maximizing expected points."""
    result = predict_goals(home_team_name, away_team_name, data)
    home_expected_goals = result['Predicted Goals (Home)']
    away_expected_goals = result['Predicted Goals (Away)']

    max_goals = 6
    best_guess = (0, 0)
    max_expected_points = float('-inf')

    for guess_home in range(max_goals + 1):
        for guess_away in range(max_goals + 1):
            expected_points = poisson.pmf(guess_home, home_expected_goals) * poisson.pmf(guess_away, away_expected_goals)

            if expected_points > max_expected_points:
                max_expected_points = expected_points
                best_guess = (guess_home, guess_away)

    return {
        "Best Superbru Prediction": f"{best_guess[0]}-{best_guess[1]}"
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "team1" not in data or "team2" not in data:
            return jsonify({"error": "Both 'team1' and 'team2' are required"}), 400

        home_team = data["team1"]
        away_team = data["team2"]

        if home_team == away_team:
            return jsonify({"error": "Teams must be different"}), 400

        df = fetch_understat_xg_data()
        if df is None:
            return jsonify({"error": "Failed to fetch xG data"}), 500

        prediction_result = full_match_prediction(home_team, away_team, df)

        return jsonify(prediction_result)

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

def full_match_prediction(home_team_name, away_team_name, df):
    """Fetch all predictions and return a response."""
    goals = predict_goals(home_team_name, away_team_name, df)
    likely_score = most_likely_score(home_team_name, away_team_name, df)
    superbru_prediction = best_superbru_prediction(home_team_name, away_team_name, df)

    return {
        **goals,
        **likely_score,
        **superbru_prediction
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
