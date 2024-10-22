import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import calculate_points_aggregation

points_average = []
points_mode = []
points_std_off = []
points_1_2 = []
points_1_5 = []
points_2 = []
path = "C:\PHM_2023_Datadump\multi\Minirocket_1024_split"

dataframes = []
labels = []
indices = []
speed_list = []

index = 0

for x, file_name in enumerate(os.listdir(path)):
    if file_name.endswith(".csv"):
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)
        label = int(file_name.split("_")[2])
        speed_number = int(file_name.split("_")[3])
        for i in range(0, len(df)):
            indices.append(index)
            labels.append(label)
            speed_list.append(speed_number)
        dataframes.append(df)
        index += 1

df = np.concatenate(dataframes)
df = pd.DataFrame(df)
speed_list = np.array(speed_list)
labels = np.array(labels)
indices = np.array(indices)

nan_columns = np.isnan(df).any(axis=0)
df = df.loc[:, ~nan_columns]
xgb_estimator = xgb.XGBRegressor()

multivar_rocket = Pipeline(
    [("scl", StandardScaler(with_mean=False)), ("clf", RidgeCV())]
)
cv = StratifiedKFold(n_splits=5)
y_pred = cross_val_predict(multivar_rocket, df, labels, cv=cv)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 10] = 10
confusion_matrix(labels, np.round(np.array(y_pred)))
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, aggregation="mean"
)
points_average.append(points)
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, aggregation="mode"
)
points_mode.append(points)
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, uncertain_std_threshold=False
)
points_std_off.append(points)
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, std=1.2
)
points_1_2.append(points)
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, std=1.5
)
points_1_5.append(points)
points = calculate_points_aggregation(
    np.round(y_pred), labels, speed_list, indices, std=2
)
points_2.append(points)
multivar_rocket = Pipeline(
    [("scl", StandardScaler(with_mean=False)), ("clf", RidgeCV())]
)
multivar_rocket.fit(df, labels)
joblib.dump(multivar_rocket, f"C:/PHM_2023_Datadump/model/solo_split/multi.joblib")
ridge_classifier = multivar_rocket.named_steps["clf"]
coefficients = ridge_classifier.coef_
