import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import clean_shipments

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

config = pd.read_csv('outputs/part_c_final_config.csv')
df = pd.read_csv('data/modelling_table.csv')

X,y = df[FEATURES], df['is_delayed']

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

if str(config['max_depth'].iloc[0]) == 'unlimited':
    depth = None
else:
    depth=int(config['max_depth'].iloc[0])

pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=int(config['selector_k'].iloc[0]))),
    ('model', RandomForestClassifier(
        n_estimators=int(config['n_estimators'].iloc[0]),
        max_depth=depth,
        min_samples_leaf=int(config['min_sample_leaf'].iloc[0]),
        random_state=42
        )
    )
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/delay_model.pkl')

pd.DataFrame({
    'threshold': [float(config['threshold'].iloc[0])]
}).to_csv(
    'models/delay_threshold.csv',
    index=False,
    float_format='%.4f'
)

pd.DataFrame({
    'feature': FEATURES
}).to_csv(
    'models/feature_order.csv',
    index=False
)

raw = pd.read_csv('data/raw_shipments.csv')
clean_df, medians = clean_shipments(raw)
impute_df = pd.DataFrame(
    list(medians.items()),
    columns=['column', 'median']
)

impute_df.to_csv(
    'models/impute_values.csv',
    index=False,
    float_format='%.4f'
)