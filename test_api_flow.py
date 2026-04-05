
import requests
import numpy as np

def test_trade_flow():
    url = "http://127.0.0.1:8000/trade"
    payload = {
        "symbol": "TATASTEEL.NS",
        "features": {"rsi_14": 30.5, "volume_ratio": 1.5},
        "current_price": 120.5,
        "observation": (np.random.randn(41)).tolist()
    }
    
    print(f"Sending POST to {url}...")
    try:
        # Note: This assumes the FastAPI server is running.
        # If it's not, this test will fail, but the user is already running it.
        r = requests.post(url, json=payload, timeout=35)
        r.raise_for_status()
        result = r.json()
        print("Success!")
        print(f"Action: {result['action']}")
        print(f"Explanation: {result['explanation'][:100]}...")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_trade_flow()
