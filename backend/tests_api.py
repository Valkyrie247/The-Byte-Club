import requests
import json

# Backend URL
URL = "http://127.0.0.1:5000/predict"

# Sample test payload (matches your dataset features)
payload = {
    "electricity_kwh": 350,
    "lpg_kg": 10,
    "fuel_litres": 50,
    "flights_year": 3,
    "public_transport": 120,
    "meat_days_week": 4,
    "waste_kg_week": 6,
    "brand_co2": 20.0
}

try:
    response = requests.post(URL, json=payload)

    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))

except Exception as e:
    print("Error:", str(e))