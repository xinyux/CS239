import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


def process_csv(read_file, col_name,dict_count):
    print ("reading file: {0} ".format(read_file))
    csv_file = pd.read_csv(read_file, low_memory=False)
    
    csv_file = csv_file[csv_file['sleep'] == 1]
    csv_file = csv_file[[col_name[0] + "_x_delta",col_name[0] + "_y_delta",col_name[0] + "_z_delta",
                         col_name[0] + "_m_delta",col_name[1] + "_x_delta",col_name[1] + "_y_delta",col_name[1] + "_z_delta",
                         col_name[1] + "_m_delta"]]
    counts = [0,0,0,0,0,0,0,0]
    cols = ['x','y','z','m']
    n = 0
    #row_count = sum(1 for row in csv_file)
    row_count = csv_file.__len__()
    print row_count
    #print col_name[0] + "_{0}_delta".format(cols[0])
    for colname in col_name:
        for col in cols:
            for i in range(0,row_count-1):
                #print csv_file.loc[csv_file.index[i],colname + "_{0}_delta".format(col)]
                if csv_file.loc[csv_file.index[i],colname + "_{0}_delta".format(col)] >20:
                    counts[n] = counts[n] + 1
            n = n + 1
            
    #count = max(counts)
    dict_count[read_file[:11]] = counts
    print counts
    return dict_count

def column(matrix, i):
    return [row[i] for row in matrix]



def main():
    interval = '300S'
    file_names = ["May_09_2017.csv","May_10_2017.csv","May_11_2017.csv","May_13_2017.csv","May_14_2017.csv",
                  "May_16_2017.csv","May_17_2017.csv","May_19_2017.csv","May_21_2017.csv","May_22_2017.csv",
                  "May_23_2017.csv","May_24_2017.csv","May_27_2017.csv",
                  "May_29_2017.csv","May_30_2017.csv","May_31_2017.csv","Jun_03_2017.csv","Jun_05_2017.csv"]
    dict_count = {}                    
    col_names = ["1_android.sensor.accelerometer","4_android.sensor.gyroscope"]
    for file_name in file_names:
        process_csv(file_name,col_names,dict_count)
    #df = pd.DataFrame({pd.Series('Date_time': index = dict_count.keys()),'times_of_movements': dict_count.values()})
    df = pd.DataFrame({'date_time': dict_count.keys(),'times_of_movements1': column(dict_count.values(),0),
                       'times_of_movements2': column(dict_count.values(),1),'times_of_movements3': column(dict_count.values(),2),
                       'times_of_movements4': column(dict_count.values(),3),'times_of_movements5': column(dict_count.values(),4),
                       'times_of_movements6': column(dict_count.values(),5),'times_of_movements7': column(dict_count.values(),6),
                       'times_of_movements8': column(dict_count.values(),7)})
    df.index = df['date_time']
    #df = pd.Series('times_of_movements': dict_count.values(), index = dict_count.keys()) 
    df['quality'] = 0
    
    df.loc['May_09_2017','quality'] = 2
    df.loc['May_10_2017','quality'] = 4
    df.loc['May_11_2017','quality'] = 3
    df.loc['May_13_2017','quality'] = 5
    df.loc['May_14_2017','quality'] = 3
    df.loc['May_16_2017','quality'] = 4
    df.loc['May_17_2017','quality'] = 3
    df.loc['May_19_2017','quality'] = 5
    df.loc['May_21_2017','quality'] = 3
    df.loc['May_22_2017','quality'] = 4
    df.loc['May_23_2017','quality'] = 2
    df.loc['May_24_2017','quality'] = 3
    #df.loc['May_25_2017','quality'] = 5
    #df.loc['May_26_2017','quality'] = 4
    df.loc['May_27_2017','quality'] = 4
    df.loc['May_29_2017','quality'] = 2
    df.loc['May_30_2017','quality'] = 3
    df.loc['May_31_2017','quality'] = 4
    df.loc['Jun_03_2017','quality'] = 2
    df.loc['Jun_05_2017','quality'] = 4
    
        
    
    df.to_csv("./sleepqualityfeature1.csv")


if __name__ == '__main__':
    main()
