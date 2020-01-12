import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = 'data'
RAW_DATA_DIR = 'wisdm_dataset/raw/watch'
COLUMNS = ['subject_id', 'activity_code', 'timestamp']

def read_sensor_data(sensor):
    if os.path.exists(os.path.join(DATA_DIR, sensor + '.csv')):
        return pd.read_csv(os.path.join(DATA_DIR, sensor + '.csv'))
    files = sorted(glob.glob(os.path.join(os.path.join(RAW_DATA_DIR, sensor), '*.txt')))
    df = pd.concat([pd.read_csv(file, header=None) for file in files], ignore_index=True)
    df[5] = df[5].str.replace(';', '').astype(float)
    df.columns = COLUMNS + [sensor + '-' + axis for axis in 'xyz']
    return df