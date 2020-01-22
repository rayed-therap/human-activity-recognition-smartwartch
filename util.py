import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = 'data'
RAW_DATA_DIR = 'wisdm-dataset/raw/watch'
ACTIVITY_CODE_PATH = 'wisdm-dataset/activity_key.txt'

# walking = 'A', jogging = 'B', stairs = 'C', sitting = 'D', standing = 'E' 
ACTIVITIES = ['A', 'B', 'C', 'D', 'E']
COLUMNS = ['subject', 'activity', 'timestamp']

def read_sensor_data(sensor):
#     if os.path.exists(os.path.join(DATA_DIR, sensor + '.csv')):
#         return pd.read_csv(os.path.join(DATA_DIR, sensor + '.csv'))
    files = sorted(glob.glob(os.path.join(os.path.join(RAW_DATA_DIR, sensor), '*.txt')))
    df = pd.concat([pd.read_csv(file, header=None) for file in files], axis=0, ignore_index=True)
#     df = df.dropna()
#     df = df.drop_duplicates()
    df[5] = df[5].str.replace(';', '', regex=False).astype(float)
    df.columns = COLUMNS + [sensor + '_' + axis for axis in 'xyz']
    df = df.loc[df['activity'].isin(ACTIVITIES)]
    df['activity'] = df['activity'].map(get_activity_map())
    return df

def get_activity_map():
    activity_map = {}
    with open(ACTIVITY_CODE_PATH) as f:
        content = f.read()
    for line in content.splitlines():
        tokens = line.split()
        if tokens:
            activity_map[tokens[2]] = tokens[0]
    return activity_map
