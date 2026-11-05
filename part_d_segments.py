import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/modelling_table.csv')
df_grouped = df.groupby('hub_id').agg(
    shipments=('shipment_id', 'nunique'),
    mean_delivery_hours=('delivery_hours', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    mean_distance_km=('distance_km','mean'),
    mean_n_stops=('n_stops', 'mean')
).reset_index()

df_grouped.to_csv('outputs/hub_profile.csv', index=False, float_format='%.4f')

sc = StandardScaler()
df_grouped = df_grouped.sort_values('hub_id').reset_index(drop=True)
df_scaled = sc.fit_transform(df_grouped[['mean_delivery_hours',
    'delay_rate',
    'mean_distance_km',
    'mean_n_stops']
    ])

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(df_scaled)

ks = [2,3,4,5]
separation = {
    'K': ks,
    'across_over_within': [],
    'cluster_sizes': []
}

for k in ks:
    km_experiment = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km_experiment.fit_predict(df_scaled)

    within, across = [], []

    for i in range(len(df_scaled)):
        for j in range(i+1, len(df_scaled)):
            distance = np.sqrt(np.sum((df_scaled[i]-df_scaled[j])**2))

            if km_labels[i] == km_labels[j]:
                within.append(distance)
            else:
                across.append(distance)
    separation['across_over_within'].append(np.mean(across)/np.mean(within))
    counts = sorted(np.bincount(km_labels), reverse=True)
    separation['cluster_sizes'].append('-'.join(map(str, counts)))

pd.DataFrame(separation).to_csv(
    'outputs/part_d_separation.csv',
    index=False,
    float_format='%.4f'
)

df_grouped['cluster'] = km.predict(df_scaled)
cluster_delay = df_grouped.groupby('cluster')['delay_rate'].mean().sort_values()
segment_names = ['Standard', 'Watch', 'Escalate']
cluster_to_segment = dict(zip(cluster_delay.index, segment_names))
df_grouped['segment'] = df_grouped['cluster'].map(cluster_to_segment)

hubs_data = pd.read_csv('data/hubs.csv')

hub_segments = df_grouped[['hub_id', 'segment']].merge(
    hubs_data[['hub_id', 'hub_city', 'hub_tier']],
    on='hub_id',
    how='left'
)

hub_segments.to_csv('outputs/hub_segments.csv', index=False)

segment_profile = df_grouped.groupby('segment').agg(
    hubs=('hub_id', 'count'),
    mean_delivery_hours=('mean_delivery_hours', 'mean'),
    delay_rate=('delay_rate', 'mean'),
    mean_distance_km=('mean_distance_km', 'mean'),
    mean_n_stops=('mean_n_stops', 'mean')
).reset_index()

segment_profile.to_csv(
    'outputs/segment_profile.csv',
    index=False,
    float_format='%.4f'
)

tier_mapping = {
    1: 'Standard',
    2: 'Watch',
    3: 'Escalate'
}

hub_segments['expected_segment'] = hub_segments['hub_tier'].map(tier_mapping)

mismatches = hub_segments[
    hub_segments['segment'] != hub_segments['expected_segment']
]

mismatches = mismatches[
    ['hub_id', 'segment', 'hub_city', 'hub_tier']
]

mismatches.to_csv('outputs/tier_vs_segment_mismatch.csv', index=False)