'''
Docstring for scripts.create_ga_csv
Reads the JSON file with the resulting matrix from the ORS API and creates a CSV file with the travel 
times in seconds and another CSV file with the travel times in minutes.
'''
import pandas as pd
import json

# Load county data into a DataFrame and the resulting matrix from the ORS API into a dictionary
counties = pd.read_csv('data/counties.csv')
with open("data/ors_matrix_result.json", "r") as f:
    result = json.load(f)

# Extract the county towns from the counties DataFrame to use as index and columns for the resulting matrix
towns = counties["county_town"].tolist()

df = pd.DataFrame(result["durations"], index=towns, columns=towns)

# Save the resulting matrix to a CSV file with travel times in seconds and another CSV file with travel times in minutes
df.to_csv('data/ga_matrix_seconds.csv')
(df/60).to_csv('data/ga_matrix_minutes.csv')