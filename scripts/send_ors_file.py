import os
from dotenv import load_dotenv
import json
import requests

load_dotenv()
api_key = os.getenv("ors_api_key")

with open("data/ors_counties.json", "r") as f:
    ors_counties = json.load(f)

url = "https://api.openrouteservice.org/v2/matrix/driving-car"

response = requests.post(
    url,
    headers={
        "Authorization": api_key,
        "Content-Type": "application/json"
    },
    json=ors_counties
)

print("Status code:", response.status_code)


with open("data/ors_matrix_result.json", "w") as f:
    json.dump(response.json(), f)

print("Matrix saved successfully")