import json 
import os 
import re
from datetime import datetime as dt
import pandas as pd
from os.path import join as pjoin
from glob import glob

path = os.getcwd()
parent = os.sep.join(path.split(os.sep)[:-1])

data_dir = rf'{parent}\data' 

print(data_dir)

years = [2021, 2022, 2023, 2024, 2025]

path = pjoin(data_dir, r'PSScene\PSScene_collection_merged.json')

collections = json.load(open(path))

regex = '(?<=\./).*?(?=\.json)'
date_list = []
for i in range(len(collections)):
    # display(i)
    for links in collections[i]['links'][1:-1]:
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

date_df.to_csv(rf'{data_dir}\PSScene\PSScene_collection_dates_new.csv', index=False)