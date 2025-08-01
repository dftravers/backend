import os
import requests
import pandas as pd
import json
import re
import logging
from scipy.stats import poisson
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import lru_cache
from datetime import datetime, timedelta
import psutil
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

CACHE_FILE = 'xg_cache.pkl'
CACHE_EXPIRY_HOURS = 3

def get_persistent_xg_data():
    """Load xG data from cache if <3h old, else scrape and update cache."""
    now = datetime.now()
    # Check if cache file exists and is fresh
    if os.path.exists(CACHE_FILE):
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if now - mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    df = pickle.load(f)
                return df
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                # Fall through to re-scrape
    # Scrape new data and cache it
    df = fetch_understat_xg_data()
    if df is not None:
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    return df

@app.route("/")
def home():
    return "Welcome to the Football Prediction API!"

def fetch_understat_xg_data():
    """Fetch xG data from Understat and calculate averages for each team."""
    try:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = "https://understat.com/league/EPL"
        logging.info(f"Starting fetch attempt at {datetime.now()}")
        
        # Reduce timeout to 10 seconds
        response = session.get(url, headers=headers, timeout=10)
        logging.info(f"Response received. Status code: {response.status_code}")
        logging.info(f"Response time: {response.elapsed.total_seconds()} seconds")
        
        # Add response headers to logs
        logging.info(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()

        raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
        if not raw_data:
            logging.error("Data pattern not found in response")
            logging.debug(f"Response content: {response.text[:500]}...")  # Log first 500 chars
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

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error when fetching xG data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching xG data: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        return None

def predict_goals(home_team_name, away_team_name, data):
    """Predict goals for each team using xG and xGA."""
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

    return home_expected_goals, away_expected_goals

def best_superbru_prediction(home_team_name, away_team_name, data):
    """Find the best Superbru prediction and the most likely score based on probabilities."""
    home_expected_goals, away_expected_goals = predict_goals(home_team_name, away_team_name, data)

    max_goals = 6
    probabilities = []
    
    most_likely_score = "0-0"
    max_prob = 0
    
    for actual_home_goals in range(max_goals + 1):
        for actual_away_goals in range(max_goals + 1):
            home_prob = poisson.pmf(actual_home_goals, home_expected_goals)
            away_prob = poisson.pmf(actual_away_goals, away_expected_goals)
            joint_prob = home_prob * away_prob
            
            probabilities.append({
                'Actual Home Goals': actual_home_goals,
                'Actual Away Goals': actual_away_goals,
                'Probability': joint_prob
            })

            if joint_prob > max_prob:
                max_prob = joint_prob
                most_likely_score = f"{actual_home_goals}-{actual_away_goals}"

    prob_df = pd.DataFrame(probabilities)

    max_expected_points = float('-inf')
    best_guess = (0, 0)
    
    for guess_home_goals in range(max_goals + 1):
        for guess_away_goals in range(max_goals + 1):
            expected_points = 0
            for _, row in prob_df.iterrows():
                actual_home = row['Actual Home Goals']
                actual_away = row['Actual Away Goals']
                probability = row['Probability']
                
                points = calculate_superbru_points(
                    guess_home_goals, guess_away_goals,
                    actual_home, actual_away
                )
                expected_points += points * probability
            
            if expected_points > max_expected_points:
                max_expected_points = expected_points
                best_guess = (guess_home_goals, guess_away_goals)
    
    return f"{best_guess[0]}-{best_guess[1]}", most_likely_score

def calculate_superbru_points(guess_home, guess_away, actual_home, actual_away):
    """Calculate Superbru points for a given guess and actual scoreline."""
    if guess_home == actual_home and guess_away == actual_away:
        return 3.0  # Exact prediction

    guess_result = "H" if guess_home > guess_away else "A" if guess_home < guess_away else "D"
    actual_result = "H" if actual_home > actual_away else "A" if actual_home < actual_away else "D"

    if abs(guess_home - actual_home) <= 1 and abs(guess_away - actual_away) <= 1:
        if guess_result == actual_result:
            return 1.5  # Close prediction points
    
    if guess_result == actual_result:
        return 1.0  # Correct result only

    return 0.0  # No points

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        log_memory_usage()
        data = request.get_json()
        if not data or "team1" not in data or "team2" not in data:
            return jsonify({"error": "Both 'team1' and 'team2' are required"}), 400

        home_team = data["team1"]
        away_team = data["team2"]

        if home_team == away_team:
            return jsonify({"error": "Teams must be different"}), 400

        df = get_persistent_xg_data()
        if df is None:
            return jsonify({"error": "Failed to fetch xG data. Please try again later."}), 503

        home_expected_goals, away_expected_goals = predict_goals(home_team, away_team, df)
        best_guess_score, most_likely_score = best_superbru_prediction(home_team, away_team, df)

        return jsonify({
            "Home Team": home_team,
            "Away Team": away_team,
            "Predicted Goals (Home)": round(home_expected_goals, 2),
            "Predicted Goals (Away)": round(away_expected_goals, 2),
            "Most Likely Score": most_likely_score,
            "Best SuperBru Prediction": best_guess_score
        })

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
