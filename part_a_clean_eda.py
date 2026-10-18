import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.shipment_prep import *

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

raw = pd.read_csv('data/raw_shipments.csv')
hubs = pd.read_csv('data/hubs.csv')

clean, medians = clean_shipments(raw)
df = build_features(clean, hubs)

df['is_delayed'] = np.where(df['delivery_hours'] > df['promised_hours'], 1,0)


df.to_csv('data/modelling_table.csv', index=False)

results = {}

raw_dropped = raw.drop_duplicates().copy()

results['raw_rows'] = len(raw)
results['duplicate_rows_removed'] = results['raw_rows'] - len(raw_dropped)
results['rows_after_dedup'] = len(raw_dropped)
results['missing_distance_km_before'] = raw_dropped['distance_km'].isna().sum()
results['missing_parcel_weight_before'] = raw_dropped['parcel_weight_kg'].isna().sum()
results['sentinel_weight_rows'] = (raw_dropped['parcel_weight_kg']>100).sum()
results['missing_declared_value_before'] = raw_dropped['declared_value'].isna().sum()
results['missing_payment_mode_before'] = raw_dropped['payment_mode'].isna().sum()
results['service_type_spellings_before'] = raw_dropped['service_type'].nunique()

results['service_type_spellings_after'] = clean['service_type'].nunique()
results['declared_value_mean_inr'] = clean['declared_value_inr'].mean()
results['log_declared_value_mean'] = df['log_declared_value'].mean()

results['median_distance_km'] = medians['distance_km']
results['median_parcel_weight_kg'] = medians['parcel_weight_kg']
results['median_declared_value_inr'] = medians['declared_value_inr']

results['rows_after_hub_merge'] = len(df)
results['nulls_in_features_and_target'] = df[FEATURES + ['delivery_hours', 'is_delayed']].isna().sum().sum()

results['delayed_count'] = (df['is_delayed'] == 1).sum()
results['delay_rate'] = (results['delayed_count']/results['rows_after_hub_merge'])

hours_q1 = df['delivery_hours'].quantile(0.25)
hours_q3 = df['delivery_hours'].quantile(0.75)

iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - (1.5*iqr)
hours_upper_bound = hours_q3 + (1.5*iqr)

results['hours_q1/hours_q3'] = str(round(hours_q1,2)) + "/" + str(round(hours_q3,2))
results['hours_lower_bound/hours_upper_bound'] = str(round(hours_lower_bound,2)) + "/" + str(round(hours_upper_bound,2))

results['hours_outlier_count'] = len(df[(df['delivery_hours'] < hours_lower_bound) | (df['delivery_hours'] > hours_upper_bound)])
results['declared_value_median_inr'] = df['declared_value_inr'].median()
results['log_declared_value_median'] = df['log_declared_value'].median()

result_df = pd.DataFrame({'metric': results.keys(), 'value': results.values()})
result_df.to_csv('data/part_a_checks.csv', columns=['metric', 'value'], index=False)

bins = [0,200,400,600,5000]
labels = ['0-200', '201-400', '401-600', '601+']
df['distance_bands'] = pd.cut(df['distance_km'], bins=bins, labels=labels, right=True)

service_summary = df.groupby('service_type').agg(
    shipments=('shipment_id', 'count'),
    delayed=('is_delayed', 'sum'),
    delay_rate=('is_delayed','mean'),
    mean_hours=('delivery_hours','mean')
)

service_summary = service_summary.reset_index()
service_summary = service_summary.rename(columns={'service_type':'group'})
service_summary['dimension'] = 'service_type'

hub_tier_summary = df.groupby('hub_tier').agg(
    shipments=('shipment_id', 'count'),
    delayed=('is_delayed', 'sum'),
    delay_rate=('is_delayed','mean'),
    mean_hours=('delivery_hours','mean')
)

hub_tier_summary = hub_tier_summary.reset_index()
hub_tier_summary = hub_tier_summary.rename(columns={'hub_tier':'group'})
hub_tier_summary['dimension'] = 'hub_tier'

distance_bands_summary = df.groupby('distance_bands').agg(
    shipments=('shipment_id', 'count'),
    delayed=('is_delayed', 'sum'),
    delay_rate=('is_delayed','mean'),
    mean_hours=('delivery_hours','mean')
)

distance_bands_summary = distance_bands_summary.reset_index()
distance_bands_summary = distance_bands_summary.rename(columns={'distance_bands':'group'})
distance_bands_summary['dimension'] = 'distance_bands'

final = pd.concat([service_summary, hub_tier_summary, distance_bands_summary], ignore_index=True)
final = final[['dimension', 'group', 'shipments', 'delayed', 'delay_rate', 'mean_hours']]
final[['delay_rate', 'mean_hours']] = final[['delay_rate', 'mean_hours']].round(4)
final.to_csv('outputs/part_a_delay_by_group.csv', index=False)

model_df = pd.read_csv('data/modelling_table.csv')
corr_matrix = model_df.select_dtypes(include='number').corr()
corr_matrix.to_csv('outputs/part_a_correlations.csv')
