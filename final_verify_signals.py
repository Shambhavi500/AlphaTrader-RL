
import requests
import os

base_url = "http://127.0.0.1:8001"

print("--- Symbol Model Prioritization Test ---")

# 1. Test GOLDBEES (We created a specific model models/GOLDBEES.NS_model.zip)
r_gold = requests.get(f"{base_url}/signal/GOLDBEES.NS")
if r_gold.status_code == 200:
    data = r_gold.json()
    print(f"GOLDBEES Signal Method: {data.get('method')}")
else:
    print(f"GOLDBEES Error: {r_gold.text}")

# 2. Test TATASTEEL (No specific model, should be TECHNICAL)
r_steel = requests.get(f"{base_url}/signal/TATASTEEL.NS")
if r_steel.status_code == 200:
    data = r_steel.json()
    print(f"TATASTEEL Signal Method: {data.get('method')}")
else:
    print(f"TATASTEEL Error: {r_steel.text}")
