import json 
import os 
import re
from datetime import datetime as dt
import pandas as pd

data_dir = r'C:\Users\phili\OneDrive\Desktop\Data Science\ADS\Spatial Statistics and Machine Learning\coral-reefs\data'

years = [2021, 2022, 2023, 2024, 2025]
collections = {}
for year in years:
    if os.path.exists(rf'{data_dir}\PSScene\PSScene_collection_{year}.json'):
        path = rf'{data_dir}\PSScene\PSScene_collection_{year}.json'
        collections[year] = json.load(open(path))

regex = '(?<=\./).*?(?=\.json)'
date_list = []
for year in collections:
    for links in collections[year]['links'][1:-1]:
        href = links['href']
        d = re.findall(regex, href)
        date_list.append(d[0])

date_df = pd.DataFrame(columns=['year', 'month', 'day', 'suffix', 'metadata'])
for i, date in enumerate(date_list):
    date_split = date.split('_')[0] + date.split('_')[1] + date.split('_')[2]
    suffix = date.split('_')[3]

    parsed_dt = dt.strptime(date_split, '%Y%m%d%H%M%S%f')

    date_df.loc[i] = {
        'year': parsed_dt.year,
        'month': parsed_dt.month,
        'day': parsed_dt.day,
        'suffix': suffix,
        'metadata': date
    }

date_df.sort_values(by=['year', 'month', 'day'], ascending=False, inplace=True)

date_df.to_csv(rf'{data_dir}\PSScene\PSScene_collection_dates.csv', index=False)