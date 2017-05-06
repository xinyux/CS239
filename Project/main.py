import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def main():
    read_file_01 = './Data/May_04_2017/1_android.sensor.accelerometer.data.csv'
    read_file_02 = './Data/May_04_2017/2_android.sensor.magnetic_field.data.csv'
    read_file_03 = './Data/May_04_2017/3_android.sensor.orientation.data.csv'
    read_file_04 = './Data/May_04_2017/4_android.sensor.gyroscope.data.csv'
    read_file_10 = './Data/May_04_2017/10_android.sensor.linear_acceleration.data.csv'

    # set downsample interval
    interval = '60S' # sample every minute
    #interval = '120S' # sample every two minutes

    # read accelerometer data, and drop last row, because last row is invalid
    print('reading file accelerometer')
    csv_file_01 = pd.read_csv(read_file_01)
    csv_file_01.columns = ["epoch_time", "x", "y", "z", "num_lable", "string_lable"]
    csv_file_01 = csv_file_01[:-1]
    csv_file_01['date_time'] = pd.to_datetime(csv_file_01['epoch_time'], unit = 'ms') + timedelta(hours=-7)
    csv_file_01.index = csv_file_01['date_time']
    csv_file_01 = csv_file_01.resample(interval, label='left').mean()

    # read magnetic field data
    print('reading file magetic field')
    csv_file_02 = pd.read_csv(read_file_02)
    csv_file_02.columns = ["epoch_time", "x", "y", "z", "num_lable", "string_lable"]
    csv_file_02 = csv_file_02[:-2]
    csv_file_02['date_time'] = pd.to_datetime(csv_file_02['epoch_time'], unit = 'ms') + timedelta(hours=-7)
    csv_file_02.index = csv_file_02['date_time']
    csv_file_02['y'] = np.float64(csv_file_02['y'])
    csv_file_02['z'] = np.float64(csv_file_02['z'])
    csv_file_02 = csv_file_02.resample(interval, label='left').mean()

    # read orientation data
    print('reading file orientation')
    csv_file_03 = pd.read_csv(read_file_03)
    csv_file_03.columns = ["epoch_time", "x", "y", "z", "num_lable", "string_lable"]
    csv_file_03 = csv_file_03[:-1]
    csv_file_03['date_time'] = pd.to_datetime(csv_file_03['epoch_time'], unit = 'ms') + timedelta(hours=-7)
    csv_file_03.index = csv_file_03['date_time']
    csv_file_03 = csv_file_03.resample(interval, label='left').mean()

    # read gyroscope data, and drop last two rows, because last two rows are invalid
    print('reading file gyroscope')
    csv_file_04 = pd.read_csv(read_file_04)
    csv_file_04 = csv_file_04[:-2]
    csv_file_04.columns = ["epoch_time", "x", "y", "z", "num_lable", "string_lable"]
    csv_file_04['date_time'] = pd.to_datetime(csv_file_04['epoch_time'], unit = 'ms') + timedelta(hours=-7)
    csv_file_04.index = csv_file_04['date_time']
    csv_file_04 = csv_file_04.resample(interval, label='left').mean()

    # read significan motion
    print('reading file linear acceleration')
    csv_file_10 = pd.read_csv(read_file_10)
    csv_file_10 = csv_file_10[:-1]
    csv_file_10.columns = ["epoch_time", "x", "y", "z", "num_lable", "string_lable"]
    csv_file_10['x'] = np.float64(csv_file_10['x'])
    csv_file_10['y'] = np.float64(csv_file_10['y'])
    csv_file_10['z'] = np.float64(csv_file_10['z'])
    csv_file_10['date_time'] = pd.to_datetime(csv_file_10['epoch_time'], unit = 'ms') + timedelta(hours=-7)
    csv_file_10.index = csv_file_10['date_time']
    csv_file_10 = csv_file_10.resample(interval, label='left').mean()

    # make plot
    print "plot accelerometer"
    df = pd.DataFrame(csv_file_01, index=csv_file_01.index, columns=list('xyz'))
    df.plot(title = "acceloerometer data");
    plt.legend()

    print "plot magnetic field"
    df = pd.DataFrame(csv_file_02, index=csv_file_02.index, columns=list('xyz'))
    df.plot(title = "magnetic data")
    plt.legend()

    print "plot orientation"
    df = pd.DataFrame(csv_file_03, index=csv_file_03.index, columns=list('xyz'))
    df.plot(title = "orientation")
    plt.legend()

    print "plot gyroscope"
    df = pd.DataFrame(csv_file_04, index=csv_file_04.index, columns=list('xyz'))
    df.plot(title = "gyroscope data")
    plt.legend()

    print "plot linear acceleration"
    df = pd.DataFrame(csv_file_10, index=csv_file_10.index, columns=list('xyz'))
    df.plot(title = "linear acceleration data")
    plt.legend()

    plt.show();


if __name__ == '__main__':
    main()