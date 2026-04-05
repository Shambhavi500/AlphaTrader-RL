
import requests
import numpy as np
import json

def verify():
    url = 'http://127.0.0.1:8001/trade'
    payload = {
        'symbol': 'TATASTEEL.NS',
        'features': {'rsi_14': 30.5, 'volume_ratio': 1.5},
        'current_price': 120.5,
        'observation': np.random.randn(41).tolist()
    }
    
    print(f"Calling {url}...")
    try:
        r = requests.post(url, json=payload, timeout=45)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Action: {data.get('action')}")
            print(f"Explanation: {data.get('explanation')}")
            return True
        else:
            print(f"Error: {r.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    verify()
