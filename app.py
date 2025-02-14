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

# ✅ Add a root route to prevent 404 errors
@app.route("/")
def home():
    return "Welcome to the Football Prediction API!"

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
    url = "https://understat.com/league/EPL"
    response = requests.get(url)
    response.raise_for_status()

    # Extract teamsData from the page
    raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
    if not raw_data:
        raise ValueError("Could not locate the teamsData variable in the page.")
    
    json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

    # Process team data
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

    # Calculate averages (avoid division by zero)
    df['Avg_xG_home'] = df['xG_home'] / df['Home_Games_Played'].replace(0, 1)
    df['Avg_xGA_home'] = df['xGA_home'] / df['Home_Games_Played'].replace(0, 1)
    df['Avg_xG_away'] = df['xG_away'] / df['Away_Games_Played'].replace(0, 1)
    df['Avg_xGA_away'] = df['xGA_away'] / df['Away_Games_Played'].replace(0, 1)

    return df

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
        'Predicted Goals (Home)': home_expected_goals,
        'Predicted Goals (Away)': away_expected_goals
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
        'Home Team': home_team_name,
        'Away Team': away_team_name,
        'Most Likely Score': f"{best_score[0]}-{best_score[1]}",
        'Probability': round(max_probability, 4)
    }

def best_superbru_prediction(home_team_name, away_team_name, data):
    """Find the best Superbru prediction by maximizing expected points."""
    result = predict_goals(home_team_name, away_team_name, data)
    home_expected_goals = result['Predicted Goals (Home)']
    away_expected_goals = result['Predicted Goals (Away)']

    max_goals = 6
    probabilities = []

    for actual_home in range(max_goals + 1):
        for actual_away in range(max_goals + 1):
            prob = poisson.pmf(actual_home, home_expected_goals) * poisson.pmf(actual_away, away_expected_goals)
            probabilities.append({'Actual Home Goals': actual_home, 'Actual Away Goals': actual_away, 'Probability': prob})

    prob_df = pd.DataFrame(probabilities)

    max_expected_points = float('-inf')
    best_guess = (0, 0)

    for guess_home in range(max_goals + 1):
        for guess_away in range(max_goals + 1):
            expected_points = sum(
                poisson.pmf(row['Actual Home Goals'], home_expected_goals) *
                poisson.pmf(row['Actual Away Goals'], away_expected_goals) *
                (3 if (guess_home == row['Actual Home Goals'] and guess_away == row['Actual Away Goals']) else 1)
                for _, row in prob_df.iterrows()
            )

            if expected_points > max_expected_points:
                max_expected_points = expected_points
                best_guess = (guess_home, guess_away)

    return {
        'Home Team': home_team_name,
        'Away Team': away_team_name,
        'Best Guess Score': f"{best_guess[0]}-{best_guess[1]}",
        'Expected Points': round(max_expected_points, 2)
    }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    home_team = data.get("team1")
    away_team = data.get("team2")
    return jsonify(full_match_prediction(home_team, away_team))

# ✅ Use environment variable for port & run in production mode
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment, default 5000
    app.run(host="0.0.0.0", port=port, debug=False)
