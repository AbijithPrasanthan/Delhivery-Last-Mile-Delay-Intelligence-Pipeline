from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
import math

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def calculate_metrics(y_actual, y_pred):
    return round(mean_squared_error(y_actual, y_pred),4), round(math.sqrt(mean_squared_error(y_actual, y_pred)),4), round(r2_score(y_actual, y_pred),4), round(mean_absolute_error(y_actual, y_pred),4)

df = pd.read_csv('data/modelling_table.csv')
X,y = df[FEATURES], df['delivery_hours']

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)
X_test_scaled = sc.transform(X_test)

checks = {"metric": [], "value": []}

checks["metric"].append('train_rows')
checks["value"].append(len(X_train))

checks['metric'].append("val_rows")
checks['value'].append(len(X_val))

checks['metric'].append('test_rows')
checks['value'].append(len(X_val))

checks['metric'].append('train_scaled_max_abs_mean')
checks['value'].append(np.abs(X_train_scaled.mean(axis=0)).max())

checks['metric'].append('test_scaled_max_abs_mean')
checks['value'].append(np.abs(X_test_scaled.mean(axis=0)).max())

checks_df = pd.DataFrame(checks)
checks_df.to_csv('outputs/part_b_checks.csv', index=False)

y_train_mean = y_train.mean()
baseline_pred = np.full(len(y_val), y_train_mean)

baseline_mse, baseline_rmse, baseline_r2, baseline_mae = calculate_metrics(y_val, baseline_pred)

depths = [1,2,3,4,6,8,12,None]
dtree_metrics = {
    'max_depth': [],
    'train_rmse': [],
    'val_rmse': []
}
best_rmse = float('inf')
best_depth = None

for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_train = dt.predict(X_train)
    y_pred_val = dt.predict(X_val)

    train_rmse = round(math.sqrt(mean_squared_error(y_train, y_pred_train)),4)
    val_rmse = round(math.sqrt(mean_squared_error(y_val, y_pred_val)),4)

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_depth = d

    if d is not None:
        dtree_metrics['max_depth'].append(d)
    else:
        dtree_metrics['max_depth'].append('unlimited')

    dtree_metrics['train_rmse'].append(train_rmse)
    dtree_metrics['val_rmse'].append(val_rmse)

dtree_metrics_df = pd.DataFrame(dtree_metrics)
dtree_metrics_df.to_csv('outputs/part_b_tree_depth.csv', index=False)

metrics = {
    'model': ['mean_baseline', 'linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting'],
    'split': ['val', 'val', 'val', 'val', 'val'],
    'MAE': [baseline_mae],
    'MSE': [baseline_mse],
    'RMSE': [baseline_rmse],
    'R2': [baseline_r2]
}

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_val)

mse, rmse, r2, mae = calculate_metrics(y_val, y_pred_lr)

metrics['MAE'].append(mae)
metrics['MSE'].append(mse)
metrics['RMSE'].append(rmse)
metrics['R2'].append(r2)

dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)

mse, rmse, r2, mae = calculate_metrics(y_val, y_pred_dt)

metrics['MAE'].append(mae)
metrics['MSE'].append(mse)
metrics['RMSE'].append(rmse)
metrics['R2'].append(r2)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

mse, rmse, r2, mae = calculate_metrics(y_val, y_pred_rf)

metrics['MAE'].append(mae)
metrics['MSE'].append(mse)
metrics['RMSE'].append(rmse)
metrics['R2'].append(r2)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_val)

mse, rmse, r2, mae = calculate_metrics(y_val, y_pred_gb)

metrics['MAE'].append(mae)
metrics['MSE'].append(mse)
metrics['RMSE'].append(rmse)
metrics['R2'].append(r2)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('outputs/part_b_metrics.csv', index=False)

learning_rates = [0.01, 0.1, 1.0]
gb_metrics = {
    'learning_rate': [],
    'val_rmse': [],
    'val_r2': []
}

fitted_models = {
    'linear_regression': lr_pipeline,
    'decision_tree': dt,
    'random_forest': rf,
    'gradient_boosting': gb
}

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
}

for lr in learning_rates:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_val)

    gb_metrics['learning_rate'].append(lr)
    gb_metrics['val_rmse'].append(round(math.sqrt(mean_squared_error(y_val, y_pred_gb)),4))
    gb_metrics['val_r2'].append(round(r2_score(y_val, y_pred_gb),4))

gb_metrics_df = pd.DataFrame(gb_metrics)
gb_metrics_df.to_csv('outputs/part_b_gb_learning.csv', index=False)

X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

candidates = metrics_df.nsmallest(2, "RMSE")["model"].to_list()
metrics_cv = {
    'model': [],
    'mean_r2': [],
    'std_r2': []
}

for candidate in candidates:
    model = models[candidate]

    scores = cross_val_score(
        model,
        X_combined,
        y_combined,
        cv=kf
    )

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    metrics_cv['model'].append(candidate)
    metrics_cv['mean_r2'].append(mean_r2)
    metrics_cv['std_r2'].append(std_r2)

cv_metrics_df = pd.DataFrame(metrics_cv)
cv_metrics_df.to_csv('outputs/part_b_cv.csv', index=False)

fin_model = fitted_models[candidates[0]]
y_pred = fin_model.predict(X_test)

mse, rmse, r2, mae = calculate_metrics(y_test, y_pred)

metrics['MAE'].append(mae)
metrics['MSE'].append(mse)
metrics['RMSE'].append(rmse)
metrics['R2'].append(r2)
metrics['model'].append(candidates[0]) 
metrics['split'].append('test')

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv('outputs/part_b_metrics.csv', index=False)
