import os
import requests
import json
import re
from datetime import datetime

# Directory to save previous seasons' data
SAVE_DIR = 'Previous Seasons'
os.makedirs(SAVE_DIR, exist_ok=True)

# Fetch Understat EPL data
url = "https://understat.com/league/EPL"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers, timeout=15)
response.raise_for_status()

raw_data = re.search(r"var teamsData = JSON.parse\('(.*?)'\);", response.text)
if not raw_data:
    raise ValueError("Could not locate the teamsData variable in the page.")

json_data = json.loads(raw_data.group(1).encode('utf-8').decode('unicode_escape'))

# Save to file with date
filename = f"epl_understat_{datetime.now().strftime('%Y%m%d')}.json"
filepath = os.path.join(SAVE_DIR, filename)
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Data saved to {filepath}") 