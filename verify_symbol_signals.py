
import requests
import json
import time

def test_symbol_models():
    # Use port 8001 since the user has a debug server running there
    base_url = "http://127.0.0.1:8001"
    
    # Symbols to test
    # GOLDBEES.NS - We created models/GOLDBEES.NS_model.zip, should be RL_MODEL
    # TATASTEEL.NS - No specific model, should be TECHNICAL
    symbols = ["GOLDBEES.NS", "TATASTEEL.NS"]
    
    print(f"--- Testing Signals on {base_url} ---")
    for symbol in symbols:
        print(f"\nRequesting signal for: {symbol}")
        try:
            r = requests.get(f"{base_url}/signal/{symbol}", timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"Action: {data.get('action')}")
                print(f"Method: {data.get('method')}")
                print(f"Confidence: {data.get('confidence')}")
            else:
                print(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_symbol_models()
