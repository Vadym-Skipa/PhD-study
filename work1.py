import os
import pandas as pd
import numpy as np
import reader_cmn as rcmn
import datetime as dt
import math

CMN_DIRECTORY = r"/home/vadymskipa/Documents/PhD_student/temp/DATA2"
SAVE_DIRECTORY = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2"


def find_all_cmn_files(directory):
    list_of_files = os.listdir(directory)
    list_of_cmn_files = []
    for file in list_of_files:
        if os.path.isfile(os.path.join(directory, file)):
            if os.path.splitext(file)[1] == ".Cmn":
                list_of_cmn_files.append(os.path.join(directory, file))
    return list_of_cmn_files


def create_pd_dataframe_from_cmn_file(file):
    dataframe_from_file = rcmn.read_cmn_file_pd(file)
    new_dataframe = pd.DataFrame({"los_tec": dataframe_from_file.loc[:, "stec"],
                                  "azm": dataframe_from_file.loc[:, "azimuth"],
                                  "elm": dataframe_from_file.loc[:, "elevation"],
                                  "gdlat": dataframe_from_file.loc[:, "latitude"],
                                  "gdlon": dataframe_from_file.loc[:, "longitude"],
                                  "time": dataframe_from_file.loc[:, "time"],
                                  "sat_id": dataframe_from_file.loc[:, "PRN"]})
    new_dataframe: pd.DataFrame = new_dataframe.loc[new_dataframe.loc[:, "time"] >= 0]
    hour_list = []
    min_list = []
    sec_list = []
    time_series = new_dataframe.loc[:, "time"]
    for index in time_series.index:
        hour = int(time_series.loc[index])
        hour_list.append(hour)
        min = int((time_series.loc[index] - hour) * 60)
        min_list.append(min)
        sec = ((time_series.loc[index] - hour) * 60 - min) * 60
        sec_list.append(sec)
        print(index)
    new_dataframe.insert(0, "hour", hour_list)
    new_dataframe.insert(0, "min", min_list)
    new_dataframe.insert(0, "sec", sec_list)
    return new_dataframe


def main():
    list_of_cmn_files = find_all_cmn_files(CMN_DIRECTORY)
    for file in list_of_cmn_files:
        print(dt.datetime.now(), f"Start reading {file}")
        dataframe: pd.DataFrame = create_pd_dataframe_from_cmn_file(file)
        dir_name = os.path.basename(file)[4:18]
        os.mkdir(os.path.join(SAVE_DIRECTORY, dir_name))
        list_of_sats = np.unique(dataframe.loc[:, "sat_id"])
        for sat_id in list_of_sats:
            local_dataframe: pd.DataFrame = dataframe.loc[dataframe.loc[:, "sat_id"] == sat_id]
            new_file = os.path.join(SAVE_DIRECTORY, dir_name, f"G{sat_id:0=2}.txt")
            with open(new_file, "w") as write_file:
                print(dt.datetime.now(), f"Start {os.path.basename(new_file)}")
                write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'los_tec':<6}\t{'azm':<6}\t{'elm':<6}"
                                 f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
                for index in local_dataframe.index:
                    print(index, "writing")
                    hour = local_dataframe.loc[index, "hour"]
                    min = local_dataframe.loc[index, "min"]
                    sec = local_dataframe.loc[index, "sec"]
                    los_tec = local_dataframe.loc[index, "los_tec"]
                    azm = local_dataframe.loc[index, "azm"]
                    elm = local_dataframe.loc[index, "elm"]
                    gdlat = local_dataframe.loc[index, "gdlat"]
                    gdlon = local_dataframe.loc[index, "gdlon"]
                    write_file.write(f"{hour:<4}\t{min:<4}\t{sec:<4.0f}\t{los_tec:<6.2f}\t{azm:<6.2f}\t{elm:<6.2f}"
                                 f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")




if __name__ == "__main__":
    main()
