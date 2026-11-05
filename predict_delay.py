import pandas as pd
import numpy as np
import joblib
from src.shipment_prep import clean_shipments, build_features

pipeline = joblib.load('models/delay_model.pkl')
new_shipments = pd.read_csv('data/new_shipments.csv')
hubs = pd.read_csv('data/hubs.csv')
delay_threshold = pd.read_csv('models/delay_threshold.csv')
feature_order = pd.read_csv('models/feature_order.csv')
impute_vals = pd.read_csv('models/impute_values.csv')
hub_segments = pd.read_csv('outputs/hub_segments.csv')

df, medians = clean_shipments(new_shipments, dict(zip(impute_vals['column'], impute_vals['median'])))
feature_names = feature_order['feature'].tolist()
featured_df = build_features(df, hubs)
X_new = featured_df[feature_names]

probs = pipeline.predict_proba(X_new)[:,1]
featured_df['delay_probability'] = np.round(probs,4)
featured_df['flagged'] = np.where((featured_df['delay_probability'] >= float(delay_threshold['threshold'].iloc[0])), 1, 0)

df_merged = featured_df.merge(
    hub_segments, on='hub_id', how='left'
)

df_merged['action'] = np.where(((df_merged['flagged'] == 1) & (df_merged['segment'] == 'Escalate')), 'Escalate to hub manager', np.where((df_merged['flagged'] == 1), 'Proactive customer call', 'No action'))

df_merged[['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']].to_csv('outputs/new_shipment_actions.csv', index=False, float_format='%0.4f')