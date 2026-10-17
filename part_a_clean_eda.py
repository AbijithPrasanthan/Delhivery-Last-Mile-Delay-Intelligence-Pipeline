import numpy as np
import pandas as pd
from shipment_prep import *

raw = pd.read_csv('data/raw_shipments.csv')
hubs = pd.read_csv('data/hubs.csv')

clean, medians = clean_shipments(raw)
df = build_features(clean, hubs)

df['is_delayed'] = np.where(df['delivery_hours'] > df['promised_hours'], 1,0)

df.to_csv('data/modelling_table.csv')