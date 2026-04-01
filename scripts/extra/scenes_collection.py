import re
import json 
import pandas as pd
from pathlib import Path
from os.path import join as pjoin
from datetime import datetime as dt

# Initialize path to data folder
parent = Path(__file__).parent.parent.parent
data_dir = pjoin(parent, r'data')


# Select json file with all the merged PSScene metadata
path = pjoin(data_dir, r'processed\metadata_merged\PSScene_collection_merged.json')

# Load the json file
collections = json.load(open(path))

# Extract the unique file names containing unique dates and suffix
regex = '(?<=\./).*?(?=\.json)'
date_list = []
for name in collections.keys():
    for i in range(len(collections[name])):
        for links in collections[name][i]['links'][1:-1]:
            href = links['href']
            d = re.findall(regex, href)
            date_list.append(f'{d[0]}_{name}') # add name for easier data wrangling

# initialize dataframe
date_df = pd.DataFrame(columns=['year', 'month', 'day', 'suffix', 'metadata', 'name'])

# Extract dates, suffix, and names into columns for manual inspection
for i, date in enumerate(date_list):
    original_metadata = date.rsplit("_", 1)[0]
    date_split = date.split('_')[0] + date.split('_')[1] + date.split('_')[2]
    suffix = date.split('_')[3]
    name = date.split('_')[4]
    parsed_dt = dt.strptime(date_split, '%Y%m%d%H%M%S%f')

    date_df.loc[i] = {
        'year': parsed_dt.year,
        'month': parsed_dt.month,
        'day': parsed_dt.day,
        'suffix': suffix,
        'metadata': original_metadata,
        'name': name,
    }

# Sort rows by year, then month, then day
date_df.sort_values(by=['year', 'month', 'day'], ascending=False, inplace=True)

# Save dataframe as csv
date_df.to_csv(rf'{data_dir}\processed\PSScene_collection_dates_new.csv', index=False)