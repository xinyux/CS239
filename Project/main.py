import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


def process_csv(read_file, interval):
    print ("reading file: " + read_file)
    csv_file = pd.read_csv(read_file, low_memory=False)
    csv_file.columns = ["epoch_time", "x", "y", "z", "num_label", "string_label"]
    csv_file = csv_file[:-5]
    csv_file['date_time'] = pd.to_datetime(csv_file['epoch_time'], unit='ms') + timedelta(hours=-7)
    csv_file.index = csv_file['date_time']
    csv_file['x'] = np.float64(csv_file['x'])
    csv_file['y'] = np.float64(csv_file['y'])
    csv_file['z'] = np.float64(csv_file['z'])

    # Select index range
    # csv_file = csv_file.loc['2017-05-07 02:30:00':'2017-05-07 04:55:00']

    # Downsampling
    csv_file = csv_file.resample(interval, label='left').mean()

    # Print data collection period
    # print (csv_file.head(1).index)
    # print (csv_file.tail(1).index)
    return csv_file


def plot_csv(csv_file, read_file):
    print ("plot file: " + read_file)
    df = pd.DataFrame(csv_file, index=csv_file.index, columns=list('xyz'))
    df.plot(title = read_file)
    plt.legend()


def main():
    interval = '60S'
    folder_names = ["May_06_2017"]
    file_names = ["1_android.sensor.accelerometer", "2_android.sensor.magnetic_field", "3_android.sensor.orientation",
                  "4_android.sensor.gyroscope", "9_android.sensor.gravity", "10_android.sensor.linear_acceleration"]
    # file_names = ["9_android.sensor.gravity"]
    for folder in folder_names:
        print ("Processing folder " + folder)
        for file in file_names:
            read_file = "./Data/" + folder + "/" + file + ".data.csv"
            csv_file = process_csv(read_file, interval)
            plot_csv(csv_file, read_file)
    plt.show()


if __name__ == '__main__':
    main()