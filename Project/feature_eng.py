import pandas as pd
import numpy as np
from datetime import timedelta


def process_csv(read_file, interval, file_name):
    print ("reading file: {0} ".format(read_file))
    csv_file = pd.read_csv(read_file, low_memory=False)
    csv_file.columns = ["epoch_time", "x", "y", "z", "num_label", "string_label"]
    csv_file['date_time'] = pd.to_datetime(csv_file['epoch_time'], unit='ms') + timedelta(hours=-7)
    csv_file.index = csv_file['date_time']
    csv_file['x'] = np.float64(csv_file['x'])
    csv_file['y'] = np.float64(csv_file['y'])
    csv_file['z'] = np.float64(csv_file['z'])
    csv_file['m'] = np.sqrt(csv_file['x']**2 + csv_file['y']**2 + csv_file['z']**2)
    csv_file = csv_file[['x','y','z','m']]

    # Downsampling
    avg_file = csv_file.resample(interval, label='left').mean()
    min_file = csv_file.resample(interval, label='left').min()
    max_file = csv_file.resample(interval, label='left').max()
    std_file = csv_file.resample(interval, label='left').std()
    avg_file.columns = [file_name + "_x_avg", file_name + "_y_avg", file_name + "_z_avg", file_name + "_m_avg"]
    min_file.columns = [file_name + "_x_min", file_name + "_y_min", file_name + "_z_min", file_name + "_m_min"]
    max_file.columns = [file_name + "_x_max", file_name + "_y_max", file_name + "_z_max", file_name + "_m_max"]
    std_file.columns = [file_name + "_x_std", file_name + "_y_std", file_name + "_z_std", file_name + "_m_std"]

    # merge files
    csv_file = pd.merge(avg_file, min_file, left_index=True, right_index=True)
    csv_file = pd.merge(csv_file, max_file, left_index=True, right_index=True)
    csv_file = pd.merge(csv_file, std_file, left_index=True, right_index=True)

    # Add a delta between max and min
    csv_file[file_name + "_x_delta"] = csv_file[file_name + "_x_max"]-csv_file[file_name + "_x_min"]
    csv_file[file_name + "_y_delta"] = csv_file[file_name + "_y_max"]-csv_file[file_name + "_y_min"]
    csv_file[file_name + "_z_delta"] = csv_file[file_name + "_z_max"]-csv_file[file_name + "_z_min"]
    csv_file[file_name + "_m_delta"] = csv_file[file_name + "_m_max"]-csv_file[file_name + "_m_min"]

    return csv_file


def feature_eng(folder_names, file_names, interval, sleep_label, train):
    print ("Start feature engineering")
    for folder in folder_names:
        print ("Processing folder {0}".format(folder))
        csv_file = pd.DataFrame()
        for file_name in file_names:
            read_file = "./Data/{0}/{1}.data.csv".format(folder, file_name)
            if csv_file.empty:
                csv_file = process_csv(read_file, interval, file_name)
            else:
                csv_file = pd.merge(csv_file, process_csv(read_file, interval, file_name), left_index=True, right_index=True)
        label_csv(csv_file, sleep_label[folder])

        # Save into LabeledData
        if train == 1:
            csv_file.to_csv("./LabeledData/{0}.csv".format(folder))

        # Save into TestData
        if train == 0:
            csv_file.to_csv("./TestData/{0}.csv".format(folder))


def label_csv(csv_file, sleep_interval):
    csv_file['sleep'] = '0'
    csv_file.loc[sleep_interval[0]:sleep_interval[1], 'sleep'] = '1'


def plot_csv(csv_file, column):
    print ("plot file: {0}".format(csv_file))
    csv_file.index = csv_file['date_time']
    df = pd.DataFrame(csv_file, index=csv_file.index, columns=column)
    df.plot()


def main():
    # Size of window to perform resampling
    interval = '300S'

    train_folder_names = ["May_09_2017","May_10_2017","May_11_2017","May_13_2017","May_14_2017", "May_16_2017", "May_17_2017",
                          "May_18_2017","May_19_2017","May_21_2017","May_22_2017","May_23_2017", "May_24_2017", "May_25_2017",
                          "May_26_2017","May_27_2017","May_29_2017","May_30_2017","May_31_2017"]
    test_folder_names = ["Jun_03_2017", "Jun_05_2017"]

    # Sensors used for feature engineering
    file_names = ["1_android.sensor.accelerometer", "2_android.sensor.magnetic_field", "3_android.sensor.orientation",
                  "4_android.sensor.gyroscope", "9_android.sensor.gravity", "10_android.sensor.linear_acceleration"]

    # Reported sleep time from participant
    sleep_label = {"May_09_2017": ['2017-05-09 01:45:00','2017-05-09 09:42:00'],
                   "May_10_2017": ['2017-05-10 00:24:00','2017-05-10 07:05:00'],
                   "May_11_2017": ['2017-05-11 01:15:00','2017-05-11 10:10:00'],
                   "May_13_2017": ['2017-05-13 02:00:00','2017-05-13 10:25:00'],
                   "May_14_2017": ['2017-05-14 01:35:00','2017-05-14 10:40:00'],
                   "May_16_2017": ['2017-05-16 01:17:00','2017-05-16 09:40:00'],
                   "May_17_2017": ['2017-05-17 00:34:00','2017-05-17 07:05:00'],
                   "May_18_2017": ['2017-05-18 03:35:00','2017-05-18 11:35:00'],
                   "May_19_2017": ['2017-05-19 01:18:00','2017-05-18 10:10:00'],
                   "May_21_2017": ['2017-05-21 01:37:00','2017-05-21 09:45:00'],
                   "May_22_2017": ['2017-05-22 00:40:00','2017-05-22 07:25:00'],
                   "May_23_2017": ['2017-05-23 01:48:00','2017-05-23 09:48:00'],
                   "May_24_2017": ['2017-05-24 00:37:00','2017-05-24 07:40:00'],
                   "May_25_2017": ['2017-05-25 00:40:00','2017-05-25 09:30:00'],
                   "May_26_2017": ['2017-05-26 01:00:00','2017-05-26 10:30:00'],
                   "May_27_2017": ['2017-05-27 02:39:00','2017-05-27 11:10:00'],
                   "May_29_2017": ['2017-05-29 01:45:00','2017-05-29 10:30:00'],
                   "May_30_2017": ['2017-05-30 01:30:00','2017-05-30 10:50:00'],
                   "May_31_2017": ['2017-05-31 01:00:00','2017-05-31 07:05:00'],
                   "Jun_03_2017": ['2017-06-03 01:40:00','2017-06-03 09:50:00'],
                   "Jun_05_2017": ['2017-06-05 00:50:00','2017-06-05 07:30:00']}

    # Create training data
    feature_eng(train_folder_names, file_names, interval, sleep_label, 1)

    # Create testing data
    feature_eng(test_folder_names, file_names, interval, sleep_label, 0)


if __name__ == '__main__':
    main()
