'''
Docstring for scripts.send_ors_file
Sends the JSON file with the locations and metrics to the ORS API and saves the resulting matrix to a new JSON file.
'''

import os
from dotenv import load_dotenv
import json
import requests

#get the api key from environment variable
load_dotenv()
api_key = os.getenv("ors_api_key")

# get the locations and metrics from the JSON file
with open("data/ors_counties.json", "r") as f:
    ors_counties = json.load(f)


#send the POST request to the ORS API and save the resulting matrix to a new JSON file
url = "https://api.openrouteservice.org/v2/matrix/driving-car"
response = requests.post(
    url,
    headers={
        "Authorization": api_key,
        "Content-Type": "application/json"
    },
    json=ors_counties
)

# save the resulting matrix to a new JSON file
with open("data/ors_matrix_result.json", "w") as f:
    json.dump(response.json(), f)