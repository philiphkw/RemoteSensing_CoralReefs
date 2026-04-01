import json 
from os.path import join as pjoin
from glob import glob
from pathlib import Path
from collections import defaultdict

# Initialize path to data folder
parent = Path(__file__).parent.parent.parent
data_dir = pjoin(parent, r'data')

# Compile all existing file paths for each metadata type
manifest_paths = glob(pjoin(data_dir, r"raw\metadata\manifest*"))
catalog_paths = glob(pjoin(data_dir, r"raw\metadata\catalog*"))
PSScene_paths = glob(pjoin(data_dir, r"raw\metadata\PSScene_collection*.json"))

# Compile manifest metadata into one dictionary
manifest_collection = {"name": "", "files": []}
for i in manifest_paths:
    file_name = i.split("\\")[-1]
    file = json.load(open(i))
    manifest_collection["files"] += file['files']

# Compile catalog metadata into one dictionary
catalog_collection = []
for i in catalog_paths:
    file = json.load(open(i))
    catalog_collection += [file]

# Compile PSScene_collections metadata into one dictionary
PSScene_merged = defaultdict(list)
for path in PSScene_paths:
    name = Path(path).stem.split("_")[-1]
    with open(path) as f:
        data = json.load(f)
    PSScene_merged[name].append(data) # Compile metadata by researcher responsible for the data
PSScene_merged = dict(PSScene_merged)

# Save the merged metadata as a json file
with open(pjoin(data_dir, r"processed\metadata_merged\manifest_merged.json"), 'w', encoding='utf-8') as f:
    json.dump(manifest_collection, f, ensure_ascii=False, indent=4)
with open(pjoin(data_dir, r"processed\metadata_merged\catalog_merged.json"), 'w', encoding='utf-8') as f:
    json.dump(catalog_collection, f, ensure_ascii=False, indent=4)
with open(pjoin(data_dir, r"processed\metadata_merged\PSScene_collection_merged.json"), 'w', encoding='utf-8') as f:
    json.dump(PSScene_merged, f, ensure_ascii=False, indent=4)