import json
import requests

# Load all required feature names
with open("feature_names.json", "r", encoding="utf-8") as f:
    features = json.load(f)

# Generate input for all 30 features
payload = {
    "features": {
        name: 1.0 for name in features
    }
}

# Call the running FastAPI endpoint
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print("Status code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))
