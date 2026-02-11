'''
Docstring for scripts.make_ors_file
Creates the JSON file with the locations and metrics for the ORS API.
'''

import pandas as pd
import json

# Load county data into a DataFrame
df = pd.read_csv('data/counties.csv')

# Extract longitude and latitude as a list of lists
locations = df[['longitude', 'latitude']].values.tolist()

# Create the dictionary for ORS API
ors_counties = {
    "locations": locations,
    "metrics": ["duration"]
}

# Save the dictionary to a JSON file
with open("data/ors_counties.json", "w") as f:
    json.dump(ors_counties, f)