import requests

url = 'http://localhost:9696/predict'

patient = {
    'age': 55.0,
    'gender': 1,
    'height': 165,
    'weight': 85.0,
    'ap_hi': 140,
    'ap_lo': 90,
    'cholesterol': 3,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1,
}

print(requests.post(url, json=patient).json())