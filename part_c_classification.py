import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier



FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def calculate_metrics_and_append(model_metrics, model, y_actual, y_pred, column_name='model'):
    cf = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cf.ravel()

    model_metrics['accuracy'].append(round(accuracy_score(y_actual, y_pred),4))
    model_metrics['precision'].append(round(precision_score(y_actual, y_pred, zero_division=0),4))
    model_metrics['recall'].append(round(recall_score(y_actual, y_pred),4))
    model_metrics['f1'].append(round(f1_score(y_actual, y_pred),4))
    model_metrics['TN'].append(tn)
    model_metrics['TP'].append(tp)
    model_metrics['FP'].append(fp)
    model_metrics['FN'].append(fn)
    model_metrics[column_name].append(model)

    return model_metrics

df = pd.read_csv('data/modelling_table.csv')
X,y = df[FEATURES], df['is_delayed']

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

print(len(X_train), len(X_test), len(X_val))
print(y_train.value_counts()[1], y_val.value_counts()[1], y_test.value_counts()[1])

model_metrics = {
    'model':[],
    'TN': [],
    'FP': [],
    'FN': [],
    'TP': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

baseline_pred = np.full(len(y_val), 0)
model_metrics = calculate_metrics_and_append(model_metrics, 'blind_all_on_time', y_val, baseline_pred)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

model_metrics = calculate_metrics_and_append(model_metrics, 'logistic_regression', y_val, y_pred)

depths = [2,3,4,5,6,8,10]
best_depth = None
best_f1 = -1

for d in depths:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_val)
    f1 = f1_score(y_val, pred)

    if f1 > best_f1:
        best_f1 = f1
        best_depth = d

dt = DecisionTreeClassifier(criterion='gini', max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_val)

model_metrics = calculate_metrics_and_append(model_metrics, 'decision_tree_depth_'+str(best_depth), y_val, y_pred)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

model_metrics = calculate_metrics_and_append(model_metrics, 'random_forest_default', y_val, y_pred)

df_metrics = pd.DataFrame(model_metrics)
df_metrics.to_csv('outputs/part_c_models.csv', index=False)

importances = rf.feature_importances_

selector = SelectKBest(f_classif, k="all")
selector.fit(X_train, y_train)

f_scores = selector.scores_

importance = pd.DataFrame({
    'feature': FEATURES,
    'f_score': f_scores,
    'rf_importance': importances,
})

importance["f_rank"] = (
    importance["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

importance["is_noise"] = importance["feature"].isin(NOISE_COLUMNS)
importance = importance[['feature', 'f_score', 'f_rank', 'rf_importance', 'is_noise']]
importance.to_csv('outputs/part_c_feature_scores.csv', index=False)

ks = [5,7,10,12,17]
selection_f1 = {
    'k': [],
    'val_f1': []
}
for k in ks:
    pipeline = Pipeline([
        ('selection', SelectKBest(f_classif, k=k)),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    selection_f1['k'].append(k)
    selection_f1['val_f1'].append(f1_score(y_val, y_pred))

pd.DataFrame(selection_f1).to_csv('outputs/part_c_selectk.csv', index=False)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}
tuning_metrics = {
    'search': ['grid_search', 'randomized_search', 'default_forest_same_7_features'],
    'candidates': [],
    'cv_folds': [],
    'total_fits': [],
    'best_params': [],
    'best_cv_accuracy': [],
    'val_f1': []
}
selector = SelectKBest(f_classif, k=7)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid.fit(X_train_selected, y_train)
candidates = len(ParameterGrid(param_grid))
grid_best_params = grid.best_params_
grid_best_cv = grid.best_score_

y_pred = grid.predict(X_val_selected)
f1_grid = f1_score(y_val, y_pred)

tuning_metrics['candidates'].append(candidates)
tuning_metrics['cv_folds'].append(grid.n_splits_)
tuning_metrics['total_fits'].append(candidates * grid.n_splits_)
tuning_metrics['best_cv_accuracy'].append(grid_best_cv)
tuning_metrics['best_params'].append(grid_best_params)
tuning_metrics['val_f1'].append(f1_grid)

random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random.fit(X_train_selected, y_train)
y_pred = random.predict(X_val_selected)

candidates = len(random.cv_results_['params'])
cv_folds = random.n_splits_
total_fits = candidates * cv_folds

f1_random = f1_score(y_val, y_pred)

tuning_metrics['candidates'].append(candidates)
tuning_metrics['cv_folds'].append(cv_folds)
tuning_metrics['total_fits'].append(total_fits)
tuning_metrics['best_cv_accuracy'].append(random.best_score_)
tuning_metrics['best_params'].append(random.best_params_)
tuning_metrics['val_f1'].append(f1_random)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train_selected, y_train)
y_pred = rf.predict(X_val_selected)
f1 = f1_score(y_val, y_pred)

tuning_metrics['candidates'].append(1)
tuning_metrics['cv_folds'].append(0)
tuning_metrics['total_fits'].append(1)
tuning_metrics['best_cv_accuracy'].append(None)
tuning_metrics['best_params'].append('defaults')
tuning_metrics['val_f1'].append(f1)

pd.DataFrame(tuning_metrics).to_csv('outputs/part_c_tuning.csv', index=False)

final_model_pipeline = Pipeline([
    ('selector', SelectKBest(f_classif, k=7)),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    ))
])

final_model_pipeline.fit(X_train, y_train)
proba = final_model_pipeline.predict_proba(X_val)[:,1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

t_metrics = {
    'threshold':[],
    'TN': [],
    'FP': [],
    'FN': [],
    'TP': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for threshold in thresholds:
    y_pred = (proba >= threshold).astype(int)
    t_metrics = calculate_metrics_and_append(t_metrics, threshold, y_val, y_pred, 'threshold')

t_df = pd.DataFrame(t_metrics)
t_df.to_csv('outputs/part_c_threshold.csv', index=False)

eligible = t_df[t_df['recall'] >= 0.80]

if not eligible.empty:
    best_threshold = eligible['threshold'].max()
else:
    best_threshold = 0.1

proba_test = final_model_pipeline.predict_proba(X_test)[:,1]
y_pred = (proba_test >= best_threshold).astype(int)

test_metrics = {
    'threshold':[],
    'TN': [],
    'FP': [],
    'FN': [],
    'TP': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

test_metrics = calculate_metrics_and_append(test_metrics, best_threshold, y_test, y_pred, 'threshold')
pd.DataFrame(test_metrics).to_csv('outputs/part_c_final_test.csv', index=False)

pd.DataFrame({
    'selector_k': [7],
    'n_estimators': [100],
    'max_depth': ['unlimited'],
    'min_samples_leaf': [1],
    'threshold': [best_threshold],
    'chosen_from': ['default_forest_same_7_features']
}).to_csv('outputs/part_c_final_config.csv', index=False)
