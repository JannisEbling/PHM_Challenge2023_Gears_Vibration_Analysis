import numpy as np
import pandas as pd


def calculate_points_aggregation(
    predictions,
    label,
    speed,
    index,
    std=1.2,
    uncertainty_if_unkown=True,
    uncertain_if_negative=True,
    uncertain_std_threshold=True,
    uncertain_value_threshold=False,
    aggregation=None,
):
    factor_check = 0
    points = 0
    prediction_probabilities, label_aggregated, speed_aggregated, indices = (
        aggregate_predictions(index, predictions, label, speed)
    )

    if aggregation == "mean":
        result = []

        for row in prediction_probabilities:
            factor = 1
            row_result = []
            for col_index, col_value in enumerate(row):
                column = [col_index] * int(col_value * 10)
                row_result.extend(column)
            result.append(row_result)
        averages = [int(np.round(sum(row) / len(row))) if row else 0 for row in result]

        points = calculate_points_simple(
            averages,
            label_aggregated,
            speed_aggregated,
            uncertainty_if_unkown=uncertainty_if_unkown,
        )

    elif aggregation == "mode":
        result = []

        for row in prediction_probabilities:
            factor = 1
            row_result = []
            for col_index, col_value in enumerate(row):
                column = [col_index] * int(col_value * 10)
                row_result.extend(column)
            result.append(row_result)
        most_occuring = [
            Counter(row).most_common(1)[0][0] if row else None for row in result
        ]
        points = calculate_points_simple(
            most_occuring,
            label_aggregated,
            speed_aggregated,
            uncertainty_if_unkown=uncertainty_if_unkown,
        )
    else:
        for i, pred_proba in enumerate(prediction_probabilities):
            factor = 1
            factor_check = 0
            points_case = 0
            if uncertainty_if_unkown == True and np.array(speed_aggregated)[i] in [
                1500,
                1800,
                2400,
            ]:
                factor = 0.2
            if i == len(prediction_probabilities) - 1:
                if (
                    np.std(predictions[indices[i] :]) > std
                    and uncertain_std_threshold == True
                ):
                    factor = 0.2
            elif (
                np.std(predictions[indices[i] : indices[i + 1]]) > std
                and uncertain_std_threshold == True
            ):
                factor = 0.2
            if uncertain_if_negative == True:
                for pseudo_label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    points_case = 0
                    for k, prediction in enumerate(pred_proba):
                        diff = abs(k - pseudo_label)

                        points_case = points_case + prediction * (1 - diff * 0.5)
                    if points_case > 0:
                        factor_check = 1
                        break
                if factor_check == 0:
                    factor = 0.2
                points_case = 0
            for j, prediction in enumerate(pred_proba):
                diff = abs(j - np.array(label_aggregated)[i])

                points_case = points_case + prediction * (1 - diff * 0.5)
            points = points + factor * points_case
    return np.round(points, 2)


def calculate_points_simple(predictions, label, speed, uncertainty_if_unkown=True):
    points = 0
    factor = 1
    for i, prediction in enumerate(predictions):
        diff = abs(np.round(prediction) - np.array(label)[i])
        if uncertainty_if_unkown == True and np.array(speed)[i] in [1500, 1800, 2400]:
            factor = 0.2
        else:
            factor = 1
        if (1 - diff * 0.5) == 1:
            points = points + factor * (1 - diff * 0.5)
        else:
            points = points + factor * (0.5 - diff * 0.5)
    return np.round(points, 1)


from collections import Counter


def aggregate_predictions(index, y_pred, label, speed):
    columns_dict = {i: [] for i in range(11)}

    current_index = None
    row_values = [0] * 11
    counter = 0
    indices = []
    # Iterate over the index and y_pred arrays
    for idx, pred in zip(index, y_pred):

        # Check if the index has changed
        if idx != current_index:
            indices.append(counter)
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
        counter += 1
    # Add the last row values to the respective columns in the dictionary
    for i, value in enumerate(row_values):
        columns_dict[i].append(value)
    # unique_index, indices = np.unique(index, return_index=True)
    label_aggregated = label[indices]
    speed_aggregated = speed[indices]
    # Create the dataframe from the dictionary
    df = pd.DataFrame(columns_dict)
    df = df.drop(0)
    df = df.div(df.sum(axis=1), axis=0).round(1)

    return np.array(df), label_aggregated, speed_aggregated, indices
