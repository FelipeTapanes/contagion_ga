import pandas as pd
import json

df = pd.read_csv('data/counties.csv')

locations = df[['longitude', 'latitude']].values.tolist()
names = df['county'].tolist()

import pandas as pd
import json

ors_counties = {
    "locations": locations,
    "metrics": ["duration"]
}


with open("data/ors_counties.json", "w") as f:
    json.dump(ors_counties, f)