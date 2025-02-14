import requests
import pandas as pd
import json
import re
from scipy.stats import poisson

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

    max_goals = 6  # Limit search space
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
        'Probability': round(max_probability, 4)  # Debugging purpose
    }

def calculate_superbru_points(guess_home, guess_away, actual_home, actual_away):
    """Calculate Superbru points for a given guess and actual scoreline."""
    if guess_home == actual_home and guess_away == actual_away:
        return 3
    
    guess_result = "H" if guess_home > guess_away else "A" if guess_home < guess_away else "D"
    actual_result = "H" if actual_home > actual_away else "A" if actual_home < actual_away else "D"
    result_points = 1 if guess_result == actual_result else 0
    
    close_points = 1 if abs(guess_home - actual_home) <= 1 and abs(guess_away - actual_away) <= 1 else 0
    
    return result_points + close_points

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
                calculate_superbru_points(guess_home, guess_away, row['Actual Home Goals'], row['Actual Away Goals'])
                * row['Probability']
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

def full_match_prediction(home_team_name, away_team_name):
    """Wrapper function that fetches data and returns all predictions."""
    df = fetch_understat_xg_data()

    goals = predict_goals(home_team_name, away_team_name, df)
    superbru = best_superbru_prediction(home_team_name, away_team_name, df)
    likely_score = most_likely_score(home_team_name, away_team_name, df)

    return {
        'Home Team': goals['Home Team'],
        'Away Team': goals['Away Team'],
        'Predicted Goals (Home)': round(goals['Predicted Goals (Home)'], 2),
        'Predicted Goals (Away)': round(goals['Predicted Goals (Away)'], 2),
        'Most Likely Score': likely_score['Most Likely Score'],
        'Best Guess Score': superbru['Best Guess Score'],
        'Expected Points': superbru['Expected Points']
    }

# Example usage:
print(full_match_prediction("Aston Villa", "Arsenal"))
