import requests
import time
import sys

def test_endpoints():
    base_url = "http://127.0.0.1:8001"
    print(f"Testing endpoints at {base_url}...")
    
    # Wait for server to start
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/api/health")
            if response.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server... ({i+1}/10)")
            time.sleep(2)
    else:
        print("Server failed to start or is not reachable.")
        return

    # Test /api/models
    print("\nTesting GET /api/models...")
    try:
        response = requests.get(f"{base_url}/api/models")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 200 and "models" in response.json():
            print("PASS: /api/models works")
        else:
            print("FAIL: /api/models failed")
    except Exception as e:
        print(f"FAIL: Error calling /api/models: {e}")

if __name__ == "__main__":
    test_endpoints()
