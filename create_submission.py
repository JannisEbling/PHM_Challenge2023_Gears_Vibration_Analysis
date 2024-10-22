import csv
import os

import joblib
import numpy as np
import pandas as pd
import torch

from models.Custom_ANN import NeuralNetwork


def get_certainty(y_pred, speed):
    certainty = 1
    if speed in [1500, 1800, 2400]:
        certainty = 0.2
    numbers = []

    for i, prob in enumerate(y_pred[0]):
        if prob >= 0.01:
            for j in range(0, int(prob * 10)):
                numbers.append(i + 1)

    std = np.std(np.array(numbers))
    if std > 0.6:
        certainty = 0.2
    factor_check = 0
    for pseudo_label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        points_case = 0
        for k, prediction in enumerate(y_pred[0]):
            diff = abs(k - pseudo_label)

            points_case = points_case + prediction * (1 - diff * 0.5)
        if points_case > 0:
            factor_check = 1
            break
    if factor_check == 0:
        certainty = 0.2
    points_case = 0
    return certainty


def aggregate_predictions_test(index, y_pred):
    columns_dict = {i: [] for i in range(11)}

    current_index = None
    row_values = [0] * 11

    # Iterate over the index and y_pred arrays
    for idx, pred in zip(index, y_pred):
        # Check if the index has changed
        if idx != current_index:
            # Add the row values to the respective columns in the dictionary
            for i, value in enumerate(row_values):
                columns_dict[i].append(value)

            # Reset the row values
            row_values = [0] * 11
            current_index = idx

        # Increment the column based on the value of y_pred
        if pred == 0:
            row_values[0] += 1
        elif pred > 0:
            column_index = int(np.round(min(pred, 10)))
            row_values[column_index] += 1
        else:
            row_values[0] += 1

    # Add the last row values to the respective columns in the dictionary
    for i, value in enumerate(row_values):
        columns_dict[i].append(value)

    df = pd.DataFrame(columns_dict)
    df = df.drop(0)
    df = df.div(df.sum(axis=1), axis=0).round(1)

    return np.array(df)


def predict_with_correct_model_split(df, speed, torque, model, model_multi):

    if speed[0] in [1500, 1800, 2400]:

        if model_multi == "reg":
            lin = joblib.load("C:/PHM_2023_Datadump/model/solo_split/multi.joblib")
        elif model_multi == "reg_filtered":
            lin = joblib.load(
                "C:/PHM_2023_Datadump/model/solo_split/multi_filtered.joblib"
            )
            bool_coef = joblib.load(
                "C:/PHM_2023_Datadump/model/solo_split/multi_bool_coef.joblib"
            )
            df = df.iloc[:, 2:]
            df = df.loc[:, bool_coef]
        elif model_multi == "ANN":
            scaler = joblib.load(
                "C:/PHM_2023_Datadump/model/solo_split/multi_scaler_ann.joblib"
            )
            connectivity_matrix = torch.zeros(700, 704)
            connectivity_matrix[:, :4] = (
                1  # Connect first two inputs to all neurons in the hidden layer
            )

            # Connect each of the remaining inputs to a single neuron in the hidden layer
            for i in range(4, 704):
                connectivity_matrix[i - 4, i] = 1
            # connectivity_matrix = connectivity_matrix.transpose(0, 1)
            # Convert the connectivity matrix to a list of lists
            print(connectivity_matrix.shape)
            connectivity_matrix = connectivity_matrix.tolist()

            lin = NeuralNetwork(704, 700, connectivity_matrix)
            lin.load_state_dict(
                torch.load(
                    "C:/PHM_2023_Datadump/model/solo_split/multi_filtered_ann.pth"
                )
            )

            bool_coef = joblib.load(
                "C:/PHM_2023_Datadump/model/solo_split/multi_bool_coef_ann.joblib"
            )
            df = df.loc[:, bool_coef]
            quadratic_df = pd.DataFrame(
                {"Column1_quadratic": df["0"] ** 2, "Column2_quadratic": df["1"] ** 2}
            )

            # Insert the new columns at positions 2 and 3
            df.insert(2, "Column1_quadratic", quadratic_df["Column1_quadratic"])
            df.insert(3, "Column2_quadratic", quadratic_df["Column2_quadratic"])
            df = scaler.transform(df)
            df = torch.tensor(df, dtype=torch.float32)
            lin.eval()
        elif model_multi == "return_0":

            y_pred = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            return y_pred, 1, None

    else:
        if model == "reg":
            lin = joblib.load(
                f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_{torque[0]}.joblib"
            )
        elif model == "reg_filtered":

            lin = joblib.load(
                f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_{torque[0]}_filtered_gaps.joblib"
            )
            bool_coef = joblib.load(
                f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_{torque[0]}_bool_coef_gaps.joblib"
            )
            df = df.iloc[:, 2:]
            df = df.loc[:, bool_coef]

        elif model == "ANN":
            scaler = joblib.load(
                f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_scaler_ann.joblib"
            )
            connectivity_matrix = torch.zeros(300, 301)
            connectivity_matrix[:, :4] = (
                1  # Connect first two inputs to all neurons in the hidden layer
            )

            # Connect each of the remaining inputs to a single neuron in the hidden layer
            for i in range(4, 301):
                connectivity_matrix[i - 4, i] = 1
            # connectivity_matrix = connectivity_matrix.transpose(0, 1)
            # Convert the connectivity matrix to a list of lists
            print(connectivity_matrix.shape)
            connectivity_matrix = connectivity_matrix.tolist()

            lin = NeuralNetwork(301, 300, connectivity_matrix)
            lin.load_state_dict(
                torch.load(
                    f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_filtered_ann.pth"
                )
            )
            bool_coef = joblib.load(
                f"C:/PHM_2023_Datadump/model/solo_split/{int(speed[0])}_bool_coef_ann.joblib"
            )
            torque = df.iloc[:, 1]
            df_new = pd.DataFrame({"torque": torque})
            df = df.iloc[:, 2:]
            df = df.loc[:, bool_coef]
            df = pd.concat([df_new, df], axis=1)
            df = scaler.transform(df)
            df = torch.tensor(df, dtype=torch.float32)
            lin.eval()
    if speed[0] in [1500, 1800, 2400] and model_multi == "ANN":
        y_pred_org = lin.forward(df)
        y_pred_org = y_pred_org.detach().numpy()

    elif speed[0] not in [1500, 1800, 2400] and model == "ANN":
        y_pred_org = lin.forward(df)
        y_pred_org = y_pred_org.detach().numpy()

    else:
        y_pred_org = lin.predict(df)
    y_pred = aggregate_predictions_test(np.zeros_like(y_pred_org), y_pred_org)
    certainty = get_certainty(y_pred, speed[0])
    return y_pred, certainty, y_pred_org


def is_sum_not_1(row):
    probabilities = [float(val) for val in row[1:-1]]
    return sum(probabilities) != 1


def adjust_probabilities(row):
    probabilities = [float(val) for val in row[1:-1]]
    total = sum(probabilities)
    if total != 1 and total > 0.01:
        max_prob_index = probabilities.index(max(probabilities))
        min_prob_index = probabilities.index(min(probabilities))
        diff = 1 - total
        probabilities[max_prob_index] += diff
        probabilities[max_prob_index] = round(probabilities[max_prob_index], 1)
        row[1:-1] = [str(prob) for prob in probabilities]
    return row


def std_calc(row):
    probabilities = [float(val) for val in row[1:-1]]
    numbers = []
    for i, prob in enumerate(probabilities):
        if prob >= 0.01:
            for j in range(0, int(prob * 10)):
                numbers.append(i + 1)

    return np.std(np.array(numbers))


all_preds = []
certainty = []

for i in np.arange(1, 801):
    data_path = f"C:\PHM_2023_Datadump/solo/Minirocket_1024/test"
    speed_list = []
    torque_list = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(f"_{i}.csv") and file_name.endswith(f""):
            speed_number = int(file_name.split("_")[0])

            torque_number = int(file_name.split("_")[1])

            file_path = os.path.join(data_path, file_name)

            df = pd.read_csv(file_path)
            df = pd.DataFrame(df)
            for j in range(0, len(df)):
                torque_list.append(torque_number)
                speed_list.append(speed_number)
            if speed_number in [1500, 1800, 2400]:
                data_path = f"C:\PHM_2023_Datadump/multi/Minirocket_1024_split/test"

            speed = np.array(speed_list)
            torque = np.array(torque_list)
            print(df.shape)

            y_pred, certainty_est, y_pred_org = predict_with_correct_model_split(
                df, speed, torque, "reg_filtered", "return_0"
            )
            all_preds.append(y_pred)
            certainty.append(certainty_est)
submission = np.concatenate(all_preds)
certainty = np.array(certainty).reshape(-1, 1)
submission = np.concatenate([submission, certainty], axis=1)


sample_ids = np.arange(1, submission.shape[0] + 1).reshape(-1, 1)

# Adding sample IDs as the first column to the array
submission = np.hstack((sample_ids, submission))

# Define the CSV file path
csv_file_path = "C:/PHM_2023_Datadump/submissions/only_unknown/submission.csv"
submission = pd.DataFrame(submission)
rows_to_adjust = submission.apply(is_sum_not_1, axis=1)
print(submission.loc[rows_to_adjust, :])
# Adjust probabilities for rows that do not sum up to 1
submission.loc[rows_to_adjust, :] = submission.loc[rows_to_adjust, :].apply(
    adjust_probabilities, axis=1
)
print(submission.loc[rows_to_adjust, :])
submission_array = submission.values
# Write the data to the CSV file with the specified header
header = [
    "sample_id",
    "prob_0",
    "prob_1",
    "prob_2",
    "prob_3",
    "prob_4",
    "prob_5",
    "prob_6",
    "prob_7",
    "prob_8",
    "prob_9",
    "prob_10",
    "confidence",
]

with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(submission_array)
