from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import json
import re
import os
from scipy.stats import poisson

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
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

def full_match_prediction(home_team_name, away_team_name):
    """Wrapper to fetch data, predict goals, and determine the best guess."""
    df = fetch_understat_xg_data()
    goals = predict_goals(home_team_name, away_team_name, df)

    return {
        'Home Team': goals['Home Team'],
        'Away Team': goals['Away Team'],
        'Predicted Goals (Home)': round(goals['Predicted Goals (Home)'], 2),
        'Predicted Goals (Away)': round(goals['Predicted Goals (Away)'], 2),
        'Best Guess Score': f"{round(goals['Predicted Goals (Home)'])}-{round(goals['Predicted Goals (Away)'])}"
    }

@app.route('/')
def home():
    """Check if the backend is running."""
    return jsonify({"message": "Backend is running!"})

@app.route('/predict', methods=['POST'])  # ✅ POST only
def predict():
    """Predict match score."""
    data = request.get_json()
    home_team = data.get('team1')
    away_team = data.get('team2')

    if not home_team or not away_team:
        return jsonify({"error": "Both teams must be selected"}), 400

    try:
        prediction_result = full_match_prediction(home_team, away_team)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
