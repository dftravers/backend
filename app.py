import os
import requests
import pandas as pd
import json
import re
import logging
from scipy.stats import poisson
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Football Prediction API!"

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team, including last update date."""
    try:
        url = "https://understat.com/league/EPL"
        response = requests.get(url)
        response.raise_for_status()

        raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
        if not raw_data:
            raise ValueError("Could not locate the teamsData variable in the page.")
        
        json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

        team_stats = []
        latest_match_date = None  # To store the most recent match date

        for team_id, team_info in json_data.items():
            home_matches = [match for match in team_info['history'] if match['h_a'] == 'h']
            away_matches = [match for match in team_info['history'] if match['h_a'] == 'a']

            # Extract match dates
            all_match_dates = [match['datetime'] for match in team_info['history']]
            if all_match_dates:
                latest_team_match = max(all_match_dates)  # Get the latest match date
                if latest_match_date is None or latest_team_match > latest_match_date:
                    latest_match_date = latest_team_match  # Update global latest match date

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

        # Convert latest match date to readable format
        if latest_match_date:
            last_updated = datetime.strptime(latest_match_date, "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")
        else:
            last_updated = "Unknown"

        return df, last_updated  # Return the dataset and last update date

    except Exception as e:
        logging.error(f"Error fetching xG data: {str(e)}")
        return None, None

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

        df, last_updated = fetch_understat_xg_data()
        if df is None or last_updated is None:
            return jsonify({"error": "Failed to fetch xG data"}), 500

        home_xg = round(df[df['Team'] == home_team]['Avg_xG_home'].values[0], 2)
        away_xg = round(df[df['Team'] == away_team]['Avg_xG_away'].values[0], 2)

        prediction_result = {
            "Predicted Goals (Home)": home_xg,
            "Predicted Goals (Away)": away_xg,
            "Last Updated": last_updated  # Include the last update timestamp
        }

        return jsonify(prediction_result)

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
