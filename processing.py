import os

import joblib
import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.series.fourier import FourierTransform


def dividedf(df, splits=3):
    df = df.iloc[:, :-1].transpose()
    num_columns = df.shape[1]
    columns_per_third = num_columns // splits
    result_df = []

    for i in range(splits):

        start_col = i * columns_per_third
        end_col = (i + 1) * columns_per_third
        third_df = df.iloc[:, start_col:end_col]
        result_df.append(np.array(third_df.transpose()))
    return result_df


def get_additional_features(split, torque, speed):
    additional_features = []
    for i, ts in enumerate(split):
        stats_df = []
        speed_torque_dict = {"speed_1": speed[i], "torque_1": torque[i]}
        speed_torque_df = pd.DataFrame.from_dict(
            speed_torque_dict, orient="index", columns=["Value"]
        )
        for i in range(ts.shape[1]):
            mean = np.mean(ts[:, i])
            std_dev = np.std(ts[:, i])
            variance = np.var(ts[:, i])
            skewness = stats.skew(ts[:, i])
            kurtosis = stats.kurtosis(ts[:, i])
            autocorr = pd.Series(ts[:, i]).autocorr(lag=1)
            frequencies, density = signal.welch(ts[:, i], nperseg=256)
            density_df = pd.DataFrame(density.transpose()).transpose()
            outlier_list = []
            for j in [
                1.5,
                1.75,
                2,
                2.25,
                2.5,
                2.75,
                3,
                3.25,
                3.5,
                3.75,
                4,
                4.25,
                4.5,
                4.75,
                5,
                5.25,
                5.5,
                5.75,
                6,
            ]:
                threshold = j * std_dev
                nmb_outliers = np.sum(np.abs(ts[:, i] - mean) > threshold)
                outlier_list.append(nmb_outliers)
            outlier_df = pd.DataFrame([outlier_list])
            stats_dict = {
                "Mean": mean,
                "Standard Deviation": std_dev,
                "Variance": variance,
                "Skewness": skewness,
                "Kurtosis": kurtosis,
                "Auto-correlation (lag 1)": autocorr,
            }
            stats_dict = (
                pd.DataFrame.from_dict(stats_dict, orient="index", columns=["0"])
                .reset_index(drop=True)
                .transpose()
            )
            density_df = density_df.reset_index(drop=True)
            stats_dict.index = density_df.index
            outlier_df.index = density_df.index
            stats_df.append(pd.concat([stats_dict, density_df, outlier_df], axis=1))
        additional_features.append(
            np.concatenate(
                [
                    np.array(speed_torque_df.transpose()),
                    np.concatenate(stats_df, axis=1),
                ],
                axis=1,
            )
        )

    # Create a DataFrame from the dictionary
    # stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Value'])
    return additional_features


def get_spectro(split, speed, split_condition=True, speedl=0):
    additional_features = []
    factor = 120 / 216
    nperseg = int((1 / ((speed / 60) * factor)) * 20480)
    g = 0

    for i, ts in enumerate(split):
        stats_df = []

        for i in range(ts.shape[1]):

            if os.path.exists("C:/PHM_2023_Datadump/fft"):
                # Load the model from file
                fft = joblib.load("C:/PHM_2023_Datadump/fft")

                fft_data = fft.transform(ts[:-1, i])
            else:
                transformer = FourierTransform()
                fft_data = transformer.fit_transform(ts[:-1, i])
                joblib.dump(transformer, "C:/PHM_2023_Datadump/fft")

            if os.path.exists("C:/PHM_2023_Datadump/mini_rocket_32_frequency2"):
                # Load the model from file
                mrm = joblib.load("C:/PHM_2023_Datadump/mini_rocket_32_frequency2")

                fft_transformed = mrm.transform(pd.DataFrame(fft_data))
            else:
                mrm = MiniRocketMultivariate(
                    num_kernels=10000, max_dilations_per_kernel=32, n_jobs=-1
                )
                fft_transformed = mrm.fit_transform(pd.DataFrame(fft_data))
                joblib.dump(mrm, "C:/PHM_2023_Datadump/mini_rocket_32_frequency2")

            stats_df.append(pd.DataFrame(fft_transformed))
        additional_features.append(np.concatenate(stats_df, axis=1))
    return additional_features


def transform(
    path, file_name, label="test", save_path="C:\PHM_2023_Datadump\solo\spectra_rocket"
):
    if label == "test":
        speed_number = int(file_name.split("_")[1][1:])
        torque_number = int(file_name.split("_")[2][:-5])
        id_number = int(file_name.split("_")[0])
        file_name_new = f"test/{speed_number}_{torque_number}_{id_number}.csv"
    else:
        speed_number = int(file_name.split("_")[0][1:])
        torque_number = int(file_name.split("_")[1][:-1])
        id_number = int(file_name.split("_")[2][:-4])
        file_name_new = f"V{speed_number}_5l/{torque_number}/{speed_number}_{torque_number}_{label}_{id_number}.csv"
    file_path = os.path.join(path, file_name)
    df = pd.read_csv(file_path, delimiter=" ", header=None)

    split_data = dividedf(df, splits=3)

    additional_features = get_spectro(split_data, speed_number)
    additional_features = pd.DataFrame(np.concatenate(additional_features))
    file_path = os.path.join(save_path, file_name_new)

    df_transformed = np.concatenate([pd.DataFrame(additional_features)], axis=1)
    # df_test.append(df_transformed)
    pd.DataFrame(df_transformed).to_csv(file_path, index=False)


save_path = "C:\PHM_2023_Datadump\spectra_rocket"
file_path = f"D:/Projects/PHM_2023/Data_Challenge_PHM2023_test_data"
not_allowed_strings = ["V1500_", "V1800_", "V2400_"]
for case in np.arange(1, 801):
    print(case)
    case_us = f"{case}_"
    for x, file_name in enumerate(os.listdir(file_path)):
        if (
            file_name.endswith(".txt")
            and file_name.startswith(f"{case_us}")
            and not any(s in file_name for s in not_allowed_strings)
        ):
            transform(file_path, file_name=file_name, save_path=save_path)

save_path = "C:\PHM_2023_Datadump\spectra_rocket"
