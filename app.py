from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import json
import re
import os
from scipy.stats import poisson
import logging

app = Flask(__name__)

# ✅ Updated CORS settings to only allow requests from your frontend
CORS(app, resources={r"/*": {"origins": ["https://dftravers.github.io", "https://www.dtravers.com"]}})

# ✅ Set up logging for debugging (useful for Render logs)
logging.basicConfig(level=logging.DEBUG)

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
    url = "https://understat.com/league/EPL"
    
    try:
        response = requests.get(url)
        logging.debug(f"Understat request status: {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Understat: {e}")
        raise ValueError("Failed to fetch xG data from Understat")

    raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
    if not raw_data:
        logging.error("Understat page structure changed or blocked request.")
        raise ValueError("Could not locate the teamsData variable in the page.")

    json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

    team_stats = []
    for team_id, team_info in json_data.items():
        home_matches = [match for match in team_info['history'] if match['h_a'] == 'h']
        away_matches = [match for match in team_info['history'] if match['h_a'] == 'a']

        team_stats.append({
            'Team': team_info['title'],
            'Home_Games_Played': len(home_matches),
            'xG_home': sum([float(match['xG']) for match in home_matches]),
            'xGA_home': sum([float(match['xGA']) for match in home_matches]),
            'Away_Games_Played': len(away_matches),
            'xG_away': sum([float(match['xG']) for match in away_matches]),
            'xGA_away': sum([float(match['xGA']) for match in away_matches]),
        })

    df = pd.DataFrame(team_stats)
    
    df['Avg_xG_home'] = df['xG_home'] / df['Home_Games_Played'].replace(0, 1)
    df['Avg_xGA_home'] = df['xGA_home'] / df['Home_Games_Played'].replace(0, 1)
    df['Avg_xG_away'] = df['xG_away'] / df['Away_Games_Played'].replace(0, 1)
    df['Avg_xGA_away'] = df['xGA_away'] / df['Away_Games_Played'].replace(0, 1)

    return df

def predict_goals(home_team_name, away_team_name, data):
    """Predict the number of goals for each team using average xG and xGA."""
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
        'Predicted Goals (Home)': home_expected_goals,
        'Predicted Goals (Away)': away_expected_goals
    }

def superbru_prediction(home_goals, away_goals):
    """Generate a more human-friendly SuperBru prediction."""
    if home_goals > away_goals:
        return "Home Win"
    elif away_goals > home_goals:
        return "Away Win"
    else:
        return "Draw"

def full_match_prediction(home_team_name, away_team_name):
    """Wrapper to fetch data, predict goals, and determine the best guess."""
    df = fetch_understat_xg_data()
    goals = predict_goals(home_team_name, away_team_name, df)

    rounded_home_goals = round(goals['Predicted Goals (Home)'], 2)
    rounded_away_goals = round(goals['Predicted Goals (Away)'], 2)
    best_guess_score = f"{round(goals['Predicted Goals (Home)'])}-{round(goals['Predicted Goals (Away)'])}"

    return {
        'Home Predicted Goals': rounded_home_goals,
        'Away Predicted Goals': rounded_away_goals,
        'Most Likely Score': best_guess_score,
        'Best SuperBru Prediction': best_guess_score  # ✅ Now same as Best Guess Score
    }


@app.route('/')
def home():
    """Check if the backend is running."""
    return jsonify({"message": "Backend is running!"})

@app.route('/predict', methods=['POST'])  # ✅ POST only
def predict():
    """Predict match score."""
    data = request.get_json()
    logging.debug(f"Received request data: {data}")

    home_team = data.get('team1')
    away_team = data.get('team2')

    if not home_team or not away_team:
        return jsonify({"error": "Both teams must be selected"}), 400

    try:
        prediction_result = full_match_prediction(home_team, away_team)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
