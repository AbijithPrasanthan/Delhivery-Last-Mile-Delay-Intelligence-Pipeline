import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

def clean_shipments(raw: pd.DataFrame, medians=None):
    df_nodup = raw.drop_duplicates().copy()
    
    df_nodup['service_type'] = df_nodup['service_type'].str.strip().str.title()
    df_nodup['payment_mode'] = df_nodup['payment_mode'].fillna("Unknown")
    df_nodup['declared_value_inr'] = pd.to_numeric(df_nodup['declared_value'].str.replace('₹','').str.replace(',',''))

    df_dropped = df_nodup.drop('declared_value', axis=1)
    df_dropped['dispatch_time'] = pd.to_datetime(df_dropped['dispatch_time'])
    
    df_dropped.loc[df_dropped['parcel_weight_kg'] > 100,'parcel_weight_kg'] = np.nan

    if not medians:
        distance_km_median = df_dropped['distance_km'].median()
        parcel_weight_median = df_dropped['parcel_weight_kg'].median()
        declared_value_median = df_dropped['declared_value_inr'].median()

        medians = {
                'distance_km': distance_km_median,
                'parcel_weight_kg': parcel_weight_median,
                'declared_value_inr': declared_value_median
            }

    else:
        distance_km_median = medians['distance_km']
        parcel_weight_median = medians['parcel_weight_kg']
        declared_value_median = medians['declared_value_inr']

    df_dropped['distance_km'] = df_dropped['distance_km'].fillna(distance_km_median)
    df_dropped['parcel_weight_kg'] = df_dropped['parcel_weight_kg'].fillna(parcel_weight_median)
    df_dropped['declared_value_inr'] = df_dropped['declared_value_inr'].fillna(declared_value_median)

    return (df_dropped,medians)

def build_features(clean, hubs):
    merged_df = pd.merge(left=clean, right=hubs, how='left', on='hub_id')
    merged_df['promised_hours'] = merged_df['service_type'].map(PROMISED_HOURS)
    merged_df['is_cod'] = np.where(merged_df['payment_mode'] == "COD", 1,0)
    merged_df['is_weekend'] = np.where((merged_df['dispatch_time'].dt.day_name() == "Saturday") | (merged_df['dispatch_time'].dt.day_name() == "Sunday"),1,0)
    merged_df['dispatch_hour'] = merged_df['dispatch_time'].dt.hour
    merged_df['log_declared_value'] = np.log(merged_df['declared_value_inr'])

    return merged_df