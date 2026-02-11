import pandas as pd
import json

counties = pd.read_csv('data/counties.csv')
with open("data/ors_matrix_result.json", "r") as f:
    result = json.load(f)

towns = counties["county_town"].tolist()

df = pd.DataFrame(result["durations"], index=towns, columns=towns)

df.to_csv('data/ga_matrix_seconds.csv')
(df/60).to_csv('data/ga_matrix_minutes.csv')