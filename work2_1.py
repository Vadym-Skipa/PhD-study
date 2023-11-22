import os
import pandas as pd
import numpy as np
import datetime as dt
from scipy.signal import savgol_filter
from typing import Dict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import matplotlib.figure as fig
import reader_cmn as rcmn
import math
from scipy.linalg import lstsq
import h5py
import read_los_file as rlos
import re

SOURCE_FILE = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G04.txt"
SOURCE_FILE2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G02.txt"
SOURCE_FILE2_1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/268-2020-09-24/G02.txt"
SOURCE_FILE3 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G08.txt"
SOURCE_FILE16 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G16.txt"

SOURCE_CMN_FILE1 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE057-2023-02-26.Cmn"
SOURCE_CMN_FILE2 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE058-2023-02-27.Cmn"
SOURCE_CMN_FILE3 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE059-2023-02-28.Cmn"
SOURCE_CMN_FILE4 = r"/home/vadymskipa/Documents/Dyplom/GPS and TEC data/267/BASE267-2020-09-23.Cmn"
SOURCE_CMN_FILE5 = r"/home/vadymskipa/Documents/Dyplom/GPS and TEC data/268/BASE268-2020-09-24.Cmn"


SAVE_PATH = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE3/"
SAVE_PATH2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE4/"
SAVE_PATH5 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE5/"
SAVE_PATH6 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/"


SOURCE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/"
SOURCE_DIRECTORY2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/268-2020-09-24/"

PERIOD_CONST = 30


# hour	min 	sec   	los_tec	azm   	elm   	gdlat 	gdlon
def read_sat_file(file=SOURCE_FILE):
    arr_hour = []
    arr_min = []
    arr_sec = []
    arr_los_tec = []
    arr_azm = []
    arr_elm = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_hour, arr_min, arr_sec, arr_los_tec, arr_azm, arr_elm, arr_gdlat, arr_gdlon]

    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append(int(text_tuple[0]))
            arr_of_arr[1].append(int(text_tuple[1]))
            arr_of_arr[2].append(int(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
            arr_of_arr[6].append(float(text_tuple[6]))
            arr_of_arr[7].append(float(text_tuple[7]))
    outcome_dataframe = pd.DataFrame({"hour": arr_hour, "min": arr_min, "sec": arr_sec, "los_tec": arr_los_tec,
                                      "azm": arr_azm, "elm": arr_elm, "gdlat": arr_gdlat, "gdlon": arr_gdlon, })
    return outcome_dataframe


def read_dtec_file(file):
    arr_hour = []
    arr_min = []
    arr_sec = []
    arr_dtec = []
    arr_azm = []
    arr_elm = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_hour, arr_min, arr_sec, arr_dtec, arr_azm, arr_elm, arr_gdlat, arr_gdlon]

    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append(int(text_tuple[0]))
            arr_of_arr[1].append(int(text_tuple[1]))
            arr_of_arr[2].append(int(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
            arr_of_arr[6].append(float(text_tuple[6]))
            arr_of_arr[7].append(float(text_tuple[7]))
    outcome_dataframe = pd.DataFrame({"hour": arr_hour, "min": arr_min, "sec": arr_sec, "dtec": arr_dtec,
                                      "azm": arr_azm, "elm": arr_elm, "gdlat": arr_gdlat, "gdlon": arr_gdlon, })
    return outcome_dataframe


# Додає до pandas dataframe, в якому є стовпчики "hour", "min", "sec", стовчик з даними "timestamp"
def add_timestamp_column_to_df(dataframe: pd.DataFrame, date: dt.datetime):
    date_timestamp = date.timestamp()
    new_dataframe = dataframe.assign(timestamp=dataframe["hour"] * 3600 + dataframe["min"] * 60 + dataframe["sec"] +
                                               date_timestamp)
    return new_dataframe


# Додає до pandas dataframe, в якому є стовпчики "timestamp", стовчик з даними "datetime"
def add_datetime_column_to_df(dataframe: pd.DataFrame):
    dataframe.loc[:, "datetime"] = pd.to_datetime(dataframe.loc[:, "timestamp"], unit="s")
    return dataframe


# Повертає List з tuples що містять стартовий та кінцевий індекс часових періодів, де:
# їхня тривалість не менша від - min_period,
# максимальна відстань між двома точками всередині періоду - max_diff_between_points
def get_time_periods(time_series: pd.Series, min_period, max_diff_between_points=30):
    diff_time_series = time_series.diff()
    breaking_index_array = diff_time_series.loc[diff_time_series > max_diff_between_points].index
    time_period_list = []
    if not len(breaking_index_array):
        time_period_list.append((time_series.index[0], time_series.index[-1]))
    else:
        start_point = time_series.index[0]
        for index in breaking_index_array:
            time_period_list.append((start_point, index - 1))
            start_point = index
        time_period_list.append((start_point, time_series.index[-1]))
    del_list = []
    for period in time_period_list:
        if (time_series.loc[period[1]] - time_series.loc[period[0]]) < min_period:
            del_list.append(period)
    if del_list:
        for period in del_list:
            time_period_list.remove(period)
    return time_period_list


def add_savgol_data_simple(dataframe: pd.DataFrame, window_length=3600, polyorder=2):
    savgol_data = savgol_filter(dataframe.loc[:, "los_tec"], (window_length // PERIOD_CONST + 1), polyorder)
    diff_data = dataframe.loc[:, "los_tec"] - savgol_data
    temp_window_length = window_length // PERIOD_CONST // 2
    new_dataframe = dataframe.assign(savgol=savgol_data,
                                     diff=diff_data.iloc[temp_window_length:-temp_window_length])
    return new_dataframe


#
# def calculate_savgol_diff_data(dataframe: pd.DataFrame, params: Dict):
#     savgov_data = savgol_filter(dataframe.loc[:, "los_tec"], **params)
#     diff_data = dataframe.loc[:, "los_tec"] - savgov_data
#     temp_window_length = (params["window_length"] - 1) // 2
#     new_dataframe = dataframe.assign(savgol=savgol_data,
#                                      diff=diff_data.iloc[temp_window_length:-temp_window_length])
#     return new_dataframe


def fill_small_gaps(dataframe: pd.DataFrame):
    time_series = dataframe.loc[:, "timestamp"]
    diff_time_series = time_series.diff()
    breaking_index_array = diff_time_series.loc[diff_time_series.loc[:] == 2 * PERIOD_CONST].index
    for index in breaking_index_array:
        new_line: pd.Series = dataframe.loc[index - 1:index].mean(0)
        date = dt.datetime.fromtimestamp(new_line.loc["timestamp"], tz=dt.timezone.utc)
        hour = date.hour
        minute = date.minute
        second = date.second
        new_line.loc["hour"] = hour
        new_line.loc["min"] = minute
        new_line.loc["sec"] = second
        dataframe = pd.concat([dataframe, new_line.to_frame().T], ignore_index=True)
    dataframe.sort_values(by="timestamp", inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def add_savgol_data_complicate(dataframe: pd.DataFrame, params: Dict, filling=False):
    if filling:
        dataframe = fill_small_gaps(dataframe)
    time_periods = get_time_periods(dataframe.loc[:, "timestamp"], (params["window_length"]))
    new_dataframe = pd.DataFrame()
    for period in time_periods:
        temp_dataframe = dataframe.loc[period[0]:period[1]]
        new_temp_dataframe = add_savgol_data(temp_dataframe, **params)
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe])
    return new_dataframe


def get_ready_dataframe(file, params: Dict, date=dt.datetime.min, filling=False):
    first_dataframe = read_sat_file(file)
    second_dataframe = add_timestamp_column_to_df(first_dataframe, date)
    third_dataframe = add_savgol_data_complicate(second_dataframe, params, filling)
    return third_dataframe


def get_dataframe_with_timestamp(file, date=dt.datetime.min):
    first_dataframe = read_sat_file(file)
    second_dataframe = add_timestamp_column_to_df(first_dataframe, date)
    return second_dataframe


def plot(save_path, name, dataframe, title=None):
    # title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("STEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    line1, = axes1.plot(time_array, dataframe["los_tec"], label="Data from GPS-TEC", linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)

    line2, = axes1.plot(time_array, dataframe["savgol"], label="Data passed through Savitzki-Golay filter",
                        linestyle=" ", marker=".", color="red", markeredgewidth=1, markersize=1.1)
    axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
    axes1.legend()
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line4, = axes2.plot(time_array, dataframe["diff"], linestyle=" ", marker=".",
                        markeredgewidth=1,
                        markersize=1,
                        label="Difference between Madrigal data and GPS-TEC")
    axes2.legend()
    axes2.set_ylim(-1.2, 1.2)
    axes2.set_xlim(*axes1.get_xlim())
    axes2.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


#
# UNREADY
# UNREADY
#
def plot_difference_graph(save_path, name, dataframe, title=None):
    if not title:
        title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    line4, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["dtec"], linestyle="-", marker=" ",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_ylim(-1.1, 1.1)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


def main_data_processing(dataframe, data_save_directory_path, window_length=3600, polyorder=2):
    list_of_params_for_processing = [{"window_length": 3600}, {"window_length": 7200}]


# Шукає файли формату Cmn в папці, але не у внутрішніх папках
def find_all_cmn_files(directory):
    list_of_files = os.listdir(directory)
    list_of_cmn_files = []
    for file in list_of_files:
        if os.path.isfile(os.path.join(directory, file)):
            if os.path.splitext(file)[1] == ".Cmn":
                list_of_cmn_files.append(os.path.join(directory, file))
    return list_of_cmn_files


# Створює pandas dataframe на основі даних Cmn файлу
# Назви стовпчики: "los_tec", "azm", "elm", "gdlat", "gdlon", "time", "sat_id", "hour", "min", "sec"
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
        sec = int(((time_series.loc[index] - hour) * 60 - min) * 60 + 0.5)
        sec_list.append(sec)
    new_dataframe.insert(0, "hour", hour_list)
    new_dataframe.insert(0, "min", min_list)
    new_dataframe.insert(0, "sec", sec_list)
    return new_dataframe


# Перетворює dataframe на інший з чітко визначеним періодом між точками та стартом відліку
# НОВІ ЧАСОВИХ КООРДИНАТ ПРИ ПЕРІОДІ 30 СЕКУНД: 00:00:00, 00:00:30, 00:01:00, 00:01:30, 00:02:00, 00:02:30, 00:03:00
# Вхідний Dataframe має стовчики "los_tec", "azm", "elm", "gdlat", "gdlon", "timestamp"
# Вхідний Dataframe є лише даними для одного супутника
def convert_dataframe_to_hard_period(dataframe: pd.DataFrame, max_nonbreakable_period_between_points=PERIOD_CONST):
    series_timestamp = dataframe.loc[:, "timestamp"]
    list_time_periods = get_time_periods(series_timestamp, PERIOD_CONST, max_nonbreakable_period_between_points)
    new_dataframe = pd.DataFrame()
    for start, finish in list_time_periods:
        temp_dataframe = dataframe.loc[start:finish]
        date = dt.datetime.fromtimestamp(temp_dataframe.iloc[0].loc["timestamp"], tz=dt.timezone.utc)
        date = dt.datetime(year=date.year, month=date.month, day=date.day, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        new_start = int((temp_dataframe.iloc[0].loc["timestamp"] - date.timestamp()) // PERIOD_CONST * PERIOD_CONST + \
                         date.timestamp() + PERIOD_CONST)
        list_new_timestamp = list(range(new_start, int(temp_dataframe.iloc[-1].loc["timestamp"]) + 1, PERIOD_CONST))
        list_new_los_tec = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                     temp_dataframe.loc[:, "los_tec"])
        list_new_azm = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"], temp_dataframe.loc[:, "azm"])
        list_new_elm = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"], temp_dataframe.loc[:, "elm"])
        list_new_gdlat = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                   temp_dataframe.loc[:, "gdlat"])
        list_new_gdlon = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                   temp_dataframe.loc[:, "gdlon"])
        new_temp_dataframe = pd.DataFrame(data={"timestamp": list_new_timestamp, "los_tec": list_new_los_tec,
                                                "azm": list_new_azm, "elm": list_new_elm, "gdlat": list_new_gdlat,
                                                "gdlon": list_new_gdlon})
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe], ignore_index=True)
    return new_dataframe


def test_convert_dataframe_to_hard_period():
    test_file = SOURCE_CMN_FILE1
    dataframe_cmn_file = create_pd_dataframe_from_cmn_file(test_file)
    cmn_file = os.path.basename(test_file)
    temp_date = dt.datetime(year=int(cmn_file[8:12]), month=int(cmn_file[13:15]),
                            day=int(cmn_file[16:18]), tzinfo=dt.timezone.utc)
    dataframe_cmn_file = add_timestamp_column_to_df(dataframe_cmn_file, temp_date)
    dataframe_cmn_file = add_datetime_column_to_df(dataframe_cmn_file)
    list_sad_id = np.unique(dataframe_cmn_file.loc[:, "sat_id"])
    for sat_id in list_sad_id:
    # sat_id = 11
        print(sat_id, dt.datetime.now())
        dataframe_cmn_sat_id = dataframe_cmn_file.loc[dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
        new_dataframe_sat_id = convert_dataframe_to_hard_period(dataframe_cmn_sat_id,
                                                                max_nonbreakable_period_between_points=3 * PERIOD_CONST)
        new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)

        for column in ["los_tec", "azm", "elm", "gdlat", "gdlon"]:
            # PLOTTING
            name = os.path.basename(test_file)
            title = column + "_" + name
            figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 320 / 300, 9.0 * 320 / 300])
            axes1: axs.Axes = figure.add_subplot(1, 1, 1)
            ytext1 = axes1.set_ylabel("column")
            xtext1 = axes1.set_xlabel("time")
            # time_array = dataframe.loc[:, "timestamp"] / 3600
            time_array = new_dataframe_sat_id.loc[:, "datetime"]
            time_array2 = dataframe_cmn_sat_id.loc[:, "datetime"]
            line1, = axes1.plot(time_array, new_dataframe_sat_id[column], label="NEW(interpolated)", color="red",
                                linestyle="-", marker=" ", markeredgewidth=0.05, markersize=0.1, linewidth=0.2)

            line2, = axes1.plot(time_array2, dataframe_cmn_sat_id[column], label="OLD(form Cmn)", linestyle=" ",
                                marker=".", color="blue", markeredgewidth=0.4, markersize=0.5)
            axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
            # axes1.set_xlim(dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=3, second=0),
            #                dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=5, second=0))
            # axes1.set_ylim(158, 160)
            axes1.legend()
            axes1.grid(True)
            save_path = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_convert_dataframe_to_hard_period/"
            plt.savefig(os.path.join(save_path, os.path.splitext(name)[0] + "_" + column + f"G{sat_id:0=2}" + ".svg"), dpi=300)
            plt.close(figure)
            # PLOTTING


def add_dataframe_with_leveling(base_dataframe: pd.DataFrame, adding_dataframe: pd.DataFrame, max_difference,
                                number_of_points=2):
    if adding_dataframe.iloc[0].loc["timestamp"] - base_dataframe.iloc[-1].loc["timestamp"] < max_difference:
        flag = True
        if base_dataframe.iloc[-1].loc["timestamp"] - base_dataframe.iloc[-number_of_points].loc["timestamp"] >= (max_difference * (number_of_points - 1)):
            indexes = base_dataframe.iloc[-number_of_points:].index
            base_dataframe.drop(indexes)
            flag = False
        if adding_dataframe.iloc[number_of_points-1].loc["timestamp"] - adding_dataframe.iloc[0].loc["timestamp"] >= (max_difference * (number_of_points - 1)):
            indexes = adding_dataframe.iloc[:number_of_points].index
            base_dataframe.drop(indexes)
            flag = False
        if flag:
            x_temp_base = np.array(base_dataframe.iloc[-number_of_points:].loc[:, "timestamp"])
            y_base = np.array(base_dataframe.iloc[-number_of_points:].loc[:, "los_tec"])
            x_base = x_temp_base[:, np.newaxis]**[0, 1]
            solve_base = lstsq(x_base, y_base)
            x_temp_adding = np.array(adding_dataframe.iloc[0: number_of_points].loc[:, "timestamp"])
            y_adding = np.array(adding_dataframe.iloc[0: number_of_points].loc[:, "los_tec"])
            x_adding = x_temp_adding[:, np.newaxis] ** [0, 1]
            solve_adding = lstsq(x_adding, y_adding)
            midpoint = (adding_dataframe.iloc[0].loc["timestamp"] + base_dataframe.iloc[-1].loc["timestamp"]) / 2
            y_base_midpoint = solve_base[0][0] + solve_base[0][1] * midpoint
            y_adding_midpoint = solve_adding[0][0] + solve_adding[0][1] * midpoint
            level_difference = y_base_midpoint - y_adding_midpoint
            adding_dataframe.loc[:, "los_tec"] = adding_dataframe.loc[:, "los_tec"] + level_difference
            list_new_timestamp = list(range(int(base_dataframe.iloc[-1].loc["timestamp"]),
                                            int(adding_dataframe.iloc[0].loc["timestamp"]), PERIOD_CONST))
            list_new_los_tec = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                              adding_dataframe.iloc[0].loc["timestamp"]],
                                         [base_dataframe.iloc[-1].loc["los_tec"],
                                          adding_dataframe.iloc[0].loc["los_tec"]])
            list_new_azm = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                              adding_dataframe.iloc[0].loc["timestamp"]],
                                         [base_dataframe.iloc[-1].loc["azm"],
                                          adding_dataframe.iloc[0].loc["azm"]])
            list_new_elm = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                              adding_dataframe.iloc[0].loc["timestamp"]],
                                         [base_dataframe.iloc[-1].loc["elm"],
                                          adding_dataframe.iloc[0].loc["elm"]])
            list_new_gdlat = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                              adding_dataframe.iloc[0].loc["timestamp"]],
                                         [base_dataframe.iloc[-1].loc["gdlat"],
                                          adding_dataframe.iloc[0].loc["gdlat"]])
            list_new_gdlon = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                              adding_dataframe.iloc[0].loc["timestamp"]],
                                         [base_dataframe.iloc[-1].loc["gdlon"],
                                          adding_dataframe.iloc[0].loc["gdlon"]])
            list_new_datetime = [dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc) for timestamp in list_new_timestamp]
            new_small_dataframe = pd.DataFrame(data={"timestamp": list_new_timestamp, "los_tec": list_new_los_tec,
                                                    "azm": list_new_azm, "elm": list_new_elm, "gdlat": list_new_gdlat,
                                                    "gdlon": list_new_gdlon, "datetime": list_new_datetime})
            base_dataframe = pd.concat([base_dataframe, new_small_dataframe], ignore_index=True)
    new_dataframe = pd.concat([base_dataframe, adding_dataframe], ignore_index=True)
    return new_dataframe


def test_add_dataframe_with_leveling(source_directory_path):
    # list_cmn_files = find_all_cmn_files(directory=source_directory_path)
    # list_cmn_files.sort()
    # list_params = [{"window_length": 3600}, {"window_length": 7200}]
    list_cmn_files = [SOURCE_CMN_FILE1, SOURCE_CMN_FILE2, SOURCE_CMN_FILE3]
    list_params = [{"window_length": 3600}]
    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    for params in list_params:
        dict_dataframe_sat_id = {}
        for cmn_file in list_cmn_files:
            temp_dataframe_cmn_file = create_pd_dataframe_from_cmn_file(cmn_file)
            file_name = os.path.basename(cmn_file)
            temp_date = dt.datetime(year=int(file_name[8:12]), month=int(file_name[13:15]),
                                    day=int(file_name[16:18]), tzinfo=dt.timezone.utc)
            temp_dataframe_cmn_file = add_timestamp_column_to_df(temp_dataframe_cmn_file, temp_date)
            list_sad_id = np.unique(temp_dataframe_cmn_file.loc[:, "sat_id"])
            for sat_id in list_sad_id:
                temp_dataframe_sat_id = temp_dataframe_cmn_file.loc[temp_dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
                new_dataframe_sat_id = convert_dataframe_to_hard_period(temp_dataframe_sat_id,
                                                                        max_nonbreakable_period_between_points)
                new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)
                if sat_id in dict_dataframe_sat_id:
                    dict_dataframe_sat_id[sat_id] = add_dataframe_with_leveling(dict_dataframe_sat_id[sat_id],
                                                                                new_dataframe_sat_id, 5 * PERIOD_CONST)
                else:
                    dict_dataframe_sat_id[sat_id] = new_dataframe_sat_id

        for sat_id, dataframe_sat_id in dict_dataframe_sat_id.items():

            # PLOTTING
            name = f"G{sat_id:0=2}"
            title = f"los_tec for {name}"
            figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 320 / 300, 9.0 * 320 / 300])
            axes1: axs.Axes = figure.add_subplot(1, 1, 1)
            ytext1 = axes1.set_ylabel("column")
            xtext1 = axes1.set_xlabel("time")
            # time_array = dataframe.loc[:, "timestamp"] / 3600
            time_array = dataframe_sat_id.loc[:, "datetime"]
            line1, = axes1.plot(time_array, dataframe_sat_id["los_tec"], label="los_tec",
                                color="blue",
                                linestyle="-", marker=".", markeredgewidth=0.5, markersize=0.6, linewidth=0.2)
            ytext1 = axes1.set_ylabel("STEC, TEC units")
            axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
            # axes1.set_xlim(dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=3, second=0),
            #                dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=5, second=0))
            # axes1.set_ylim(158, 160)
            axes1.legend()
            axes1.grid(True)
            save_path = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_add_dataframe_with_leveling/"
            plt.savefig(
                os.path.join(save_path, f"G{sat_id:0=2}" + ".svg"),
                dpi=300)
            plt.close(figure)
            # PLOTTING


def get_directory_path(existing_directory_path, new_folder):
    new_path = os.path.join(existing_directory_path, new_folder)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def get_date_str(datetime: dt.datetime):
    return f"{datetime.timetuple().tm_yday:0=3}-{datetime.year}-{datetime.month:0=2}-{datetime.day:0=2}"


def add_savgol_data(dataframe: pd.DataFrame, params: Dict):
    time_periods = get_time_periods(dataframe.loc[:, "timestamp"], (params["window_length"]))
    new_dataframe = pd.DataFrame()
    for period in time_periods:
        temp_dataframe = dataframe.loc[period[0]:period[1]]
        new_temp_dataframe = add_savgol_data_simple(temp_dataframe, **params)
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe])
    return new_dataframe


def save_diff_dataframe_txt(save_file_path, dataframe: pd.DataFrame):
    dataframe_without_nan = dataframe.loc[dataframe.loc[:, "diff"].notna()]
    with open(save_file_path, "w") as write_file:
        write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dTEC':<6}\t{'azm':<6}\t{'elm':<6}"
                         f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
        for index in dataframe_without_nan.index:
            hour = dataframe_without_nan.loc[index, "datetime"].hour
            minute = dataframe_without_nan.loc[index, "datetime"].minute
            sec = dataframe_without_nan.loc[index, "datetime"].second
            dtec = dataframe_without_nan.loc[index, "diff"]
            azm = dataframe_without_nan.loc[index, "azm"]
            elm = dataframe_without_nan.loc[index, "elm"]
            gdlat = dataframe_without_nan.loc[index, "gdlat"]
            gdlon = dataframe_without_nan.loc[index, "gdlon"]
            write_file.write(f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{dtec:<6.3f}\t{azm:<6.2f}\t{elm:<6.2f}"
                             f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")


SAVE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/"
SOURCE_CMN_DIRECTORY1 = r"/home/vadymskipa/Documents/GPS_2023_outcome/"
def main1(source_directory_path, save_directory_path):
    list_cmn_files = find_all_cmn_files(directory=source_directory_path)
    list_cmn_files.sort()
    # list_cmn_files = [SOURCE_CMN_FILE4, SOURCE_CMN_FILE5]
    list_params = [{"window_length": 3600}, {"window_length": 7200}]
    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    save_path_txt = get_directory_path(save_directory_path, "TXT")
    save_path_csv = get_directory_path(save_directory_path, "CSV")
    for params in list_params:
        dict_dataframe_sat_id = {}
        for cmn_file in list_cmn_files:
            temp_dataframe_cmn_file = create_pd_dataframe_from_cmn_file(cmn_file)
            file_name = os.path.basename(cmn_file)
            temp_date = dt.datetime(year=int(file_name[8:12]), month=int(file_name[13:15]),
                                    day=int(file_name[16:18]), tzinfo=dt.timezone.utc)
            temp_dataframe_cmn_file = add_timestamp_column_to_df(temp_dataframe_cmn_file, temp_date)
            list_sad_id = np.unique(temp_dataframe_cmn_file.loc[:, "sat_id"])
            for sat_id in list_sad_id:
                temp_dataframe_sat_id = temp_dataframe_cmn_file.loc[temp_dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
                new_dataframe_sat_id = convert_dataframe_to_hard_period(temp_dataframe_sat_id,
                                                                        max_nonbreakable_period_between_points)
                new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)
                if sat_id in dict_dataframe_sat_id:
                    dict_dataframe_sat_id[sat_id] = add_dataframe_with_leveling(dict_dataframe_sat_id[sat_id],
                                                                                new_dataframe_sat_id,
                                                                                max_nonbreakable_period_between_points,
                                                                                number_of_points=4)
                else:
                    dict_dataframe_sat_id[sat_id] = new_dataframe_sat_id
        for sat_id, dataframe_sat_id in dict_dataframe_sat_id.items():
            dataframe_sat_id = add_savgol_data(dataframe_sat_id, params)
            first_timestamp = dataframe_sat_id.iloc[0].loc["timestamp"]
            first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
            first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                              day=first_datetime.day, hour=0, minute=0, second=0,
                                              tzinfo=dt.timezone.utc).timestamp()
            series_timestamp = (dataframe_sat_id.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                               (3600 * 24) + first_day_timestamp
            array_timestamp = np.unique(series_timestamp)
            for timestamp in array_timestamp:
                date = dt.datetime.fromtimestamp(timestamp)
                save_path_txt_date = get_directory_path(save_path_txt, get_date_str(date))
                save_path_csv_date = get_directory_path(save_path_csv, get_date_str(date))
                save_path_txt_params = get_directory_path(save_path_txt_date,
                                                          f"Window_{params['window_length']}_Seconds")
                save_path_csv_params = get_directory_path(save_path_csv_date,
                                                          f"Window_{params['window_length']}_Seconds")
                save_path_txt_sat_id = os.path.join(save_path_txt_params, f"G{sat_id:0=2}.txt")
                save_path_csv_sat_id = os.path.join(save_path_csv_params, f"G{sat_id:0=2}.csv")
                mask = dataframe_sat_id.loc[:, "timestamp"] >= timestamp
                mask = np.logical_and(mask, dataframe_sat_id.loc[:, "timestamp"] < (timestamp + 3600 * 24))
                temp_dataframe: pd.DataFrame = dataframe_sat_id.loc[mask]
                temp_dataframe.to_csv(save_path_csv_sat_id)
                save_diff_dataframe_txt(save_path_txt_sat_id, temp_dataframe)


SOURCE_DIRECTORY_NEW1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/2020-09-23/"
SOURCE_DIRECTORY_OLD1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/267-2020-09-23/"
SOURCE_DIRECTORY_NEW2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/2020-09-24/"
SOURCE_DIRECTORY_OLD2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/268-2020-09-24/"
SAVE_DIRECTORY_TEST1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_plot_new_savgov_vs_old_savgov/"
def test_plot_new_savgov_vs_old_savgov():
    list_directory = [(SOURCE_DIRECTORY_OLD1, SOURCE_DIRECTORY_NEW1),
                      (SOURCE_DIRECTORY_OLD2, SOURCE_DIRECTORY_NEW2)]
    for old_directory_path, new_directory_path in list_directory:
        list_inner_directory = os.listdir(old_directory_path)
        year = int(os.path.basename(os.path.dirname(old_directory_path))[4:8])
        month = int(os.path.basename(os.path.dirname(old_directory_path))[9:11])
        day = int(os.path.basename(os.path.dirname(old_directory_path))[12:14])
        date = dt.datetime(year=year, month=month, day=day, tzinfo=dt.timezone.utc)
        for inner_directory in list_inner_directory:
            list_dtec_file = os.listdir(os.path.join(old_directory_path, inner_directory))
            save_directory_path = get_directory_path(SAVE_DIRECTORY_TEST1,
                                                     os.path.basename(os.path.dirname(old_directory_path)))
            save_directory_path = get_directory_path(save_directory_path, inner_directory)
            for dtec_file in list_dtec_file:
                str_sat_id = os.path.splitext(dtec_file)[0]
                dtec_file_old_path = os.path.join(old_directory_path, inner_directory, dtec_file)
                dtec_file_new_path = os.path.join(new_directory_path, inner_directory, dtec_file)
                dataframe_old = read_dtec_file(dtec_file_old_path)
                dataframe_old = add_timestamp_column_to_df(dataframe_old, date)
                dataframe_old = add_datetime_column_to_df(dataframe_old)
                dataframe_new = read_dtec_file(dtec_file_new_path)
                dataframe_new = add_timestamp_column_to_df(dataframe_new, date)
                dataframe_new = add_datetime_column_to_df(dataframe_new)
                title = f"Comparison of two savgol approaches (without midnight and with) for {str_sat_id}" \
                        f"\n{inner_directory}, {dt.date(year=year, month=month, day=day)}"
                figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
                axes1: axs.Axes = figure.add_subplot(1, 1, 1)
                line1, = axes1.plot(dataframe_new.loc[:, "datetime"], dataframe_new["dtec"], linestyle="-", marker=" ",
                                    linewidth=0.3, color="red", label="with midnight")
                line2, = axes1.plot(dataframe_old.loc[:, "datetime"], dataframe_old["dtec"], linestyle=" ", marker=".",
                                    markersize=1.0, color="blue", label="without midnight")
                axes1.set_ylim(-1.1, 1.1)
                axes1.set_xlim(dataframe_new.iloc[0].loc["datetime"], dataframe_new.iloc[-1].loc["datetime"])
                axes1.grid(True)
                axes1.set_title(title)
                axes1.legend()
                plt.savefig(os.path.join(save_directory_path, f"{str_sat_id}.png"), dpi=300)
                plt.close(figure)
                print(str_sat_id)










def test_lstsq():
    x1 = np.array([3, 4, 5])
    y1 = np.array([5, 7.1, 5])
    x = x1[:, np.newaxis]**[0, 1]
    a = lstsq(x, y1)
    print(a)




def save_los_tec_txt_from_dataframe(save_path, dataframe: pd.DataFrame):
    with open(save_path, "w") as write_file:
        write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'los_tec':<6}\t{'azm':<6}\t{'elm':<6}"
                         f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
        for index in dataframe.index:
            hour = dataframe.loc[index, "hour"]
            min = dataframe.loc[index, "min"]
            sec = dataframe.loc[index, "sec"]
            los_tec = dataframe.loc[index, "los_tec"]
            azm = dataframe.loc[index, "azm"]
            elm = dataframe.loc[index, "elm"]
            gdlat = dataframe.loc[index, "gdlat"]
            gdlon = dataframe.loc[index, "glon"]
            write_file.write(f"{hour:<4}\t{min:<4}\t{sec:<4.0f}\t{los_tec:<6.2f}\t{azm:<6.2f}\t{elm:<6.2f}"
                             f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")


SITES1 = ['bute', 'bor1', 'fra2', 'plnd', 'polv', 'krrs', 'mikl', 'pryl', 'cfrm']
LOS_FILE_PATH1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230226.001.h5.hdf5"
SAVE_PATH7 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/"
def save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path):
    new_save_path = get_directory_path(save_path, "los_tec_txt")
    start = dt.datetime.now()
    print(f"Reading \"gps_site\" column from los_fie ------------ {start}")
    with h5py.File(los_file_path, "r") as file:
        site_array = file["Data"]["Table Layout"]["gps_site"]
    print(f"End of reading \"gps_site\" column from los_fie ----- {dt.datetime.now()} -- {dt.datetime.now()-start}")

    mask = np.full(len(site_array), False)
    for site in list_gps_sites:
        temp_mask = site_array == site.encode("ascii")
        mask = np.logical_or(mask, temp_mask)

    print(f"Reading data for sites ------------------------------- {dt.datetime.now()}")
    los_dataframe = rlos.get_data_by_indecies_GPS_pd(los_file_path, mask)
    print(f"End of reading data for sites ------------------------ {dt.datetime.now()} -- {dt.datetime.now()-start}")

    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    name_los_file = os.path.basename(los_file_path)
    date = dt.datetime(int(name_los_file[4:8]), int(name_los_file[8:10]), int(name_los_file[10:12]))
    los_dataframe = add_timestamp_column_to_df(los_dataframe, date)
    for site in list_gps_sites:
        print(f"Saving los_tec_txt for site {site} ----------------- {dt.datetime.now()} -- {dt.datetime.now()-start}")
        save_path_site_1 = get_directory_path(new_save_path, site)
        site_dataframe = los_dataframe.loc[los_dataframe.loc[:, "gps_site"] == site.encode("ascii")]
        save_path_date_2 = get_directory_path(save_path_site_1, get_date_str(date))
        sat_id_array = np.unique(site_dataframe.loc[:, "sat_id"])
        for sat_id in sat_id_array:
            save_path_sat_id_3 = os.path.join(save_path_date_2, f"G{sat_id:0=2}.txt")
            sat_id_dataframe = site_dataframe.loc[site_dataframe.loc[:, "sat_id"] == sat_id]
            save_los_tec_txt_from_dataframe(save_path_sat_id_3, sat_id_dataframe)


def get_los_file_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory_path, el)) and
                       re.search("^los_(\d{8})\S*(\.h5|\.hdf5)$", el))]
    return result_list


def get_date_directory_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isdir(os.path.join(directory_path, el)) and
                       re.search("^(\d{3})-(\d{4})-(\d{2})-(\d{2})$", el))]
    return result_list


def get_sat_id_file_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory_path, el)) and
                       re.search("^G(\d{2})\.txt$", el))]
    return result_list



DIRECTORY_PATH1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/"
def save_los_tec_txt_file_for_some_sites_from_directory_with_los_files(directory_path, list_gps_sites, save_path):
    list_los_file_paths = get_los_file_paths_from_directory(directory_path)
    for los_file_path in list_los_file_paths:
        save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path)


def get_site_directory_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isdir(os.path.join(directory_path, el)) and
                       re.search("^(\S{4,4})$", el))]
    return result_list


def get_date_from_date_directory_name(date_directory_name):
    date = dt.datetime(int(date_directory_name[4:8]), int(date_directory_name[9:11]), int(date_directory_name[12:14]),
                       tzinfo=dt.timezone.utc)
    return date



SAVE_PATH8 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/bor1/"
SOURCE_DIRECTORY_PATH8 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/los_tec_txt/bor1/"
LIST_PARAMS1 = ({"window_length": 3600}, {"window_length": 7200})
def get_dtec_from_los_tec_txt_1(source_directory_path, save_directory_path, list_params=LIST_PARAMS1):
    max_nonbreakable_period_between_points = 8 * PERIOD_CONST
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_directory_path)
    list_of_date_directory_paths.sort()
    dict_dataframe_sat_id = {}
    for date_directory_path in list_of_date_directory_paths:
        name_date_directory = os.path.basename(date_directory_path)
        date = get_date_from_date_directory_name(name_date_directory)
        list_of_sat_id_los_tec_file_paths = get_sat_id_file_paths_from_directory(date_directory_path)
        for sat_id_los_tec_file_path in list_of_sat_id_los_tec_file_paths:
            name_sat_id_file = os.path.basename(sat_id_los_tec_file_path)
            los_tec_dataframe = read_sat_file(sat_id_los_tec_file_path)
            los_tec_dataframe = add_timestamp_column_to_df(los_tec_dataframe, date)
            los_tec_dataframe = convert_dataframe_to_hard_period(los_tec_dataframe,
                                                                 max_nonbreakable_period_between_points)
            if name_sat_id_file in dict_dataframe_sat_id:
                dict_dataframe_sat_id[name_sat_id_file] = \
                    add_dataframe_with_leveling(dict_dataframe_sat_id[name_sat_id_file],
                                                los_tec_dataframe, max_nonbreakable_period_between_points,
                                                number_of_points=4)
            else:
                dict_dataframe_sat_id[name_sat_id_file] = los_tec_dataframe
    for name_sat_id_file, los_tec_dataframe in dict_dataframe_sat_id.items():
        for params in list_params:
            dataframe_sat_id = add_savgol_data(los_tec_dataframe, params)
            dataframe_sat_id = add_datetime_column_to_df(dataframe_sat_id)
            first_timestamp = dataframe_sat_id.iloc[0].loc["timestamp"]
            first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
            first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                              day=first_datetime.day, hour=0, minute=0, second=0,
                                              tzinfo=dt.timezone.utc).timestamp()
            series_timestamp = (dataframe_sat_id.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                               (3600 * 24) + first_day_timestamp
            array_timestamp = np.unique(series_timestamp)
            for timestamp in array_timestamp:
                date = dt.datetime.fromtimestamp(timestamp)
                save_path_date_1 = get_directory_path(save_directory_path, get_date_str(date))
                save_path_params_2 = get_directory_path(save_path_date_1,
                                                        f"Window_{params['window_length']}_Seconds")
                save_path_sat_id = os.path.join(save_path_params_2, name_sat_id_file)
                mask = dataframe_sat_id.loc[:, "timestamp"] >= timestamp
                mask = np.logical_and(mask, dataframe_sat_id.loc[:, "timestamp"] < (timestamp + 3600 * 24))
                temp_dataframe: pd.DataFrame = dataframe_sat_id.loc[mask]
                save_diff_dataframe_txt(save_path_sat_id, temp_dataframe)





SAVE_PATH10 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/"
SOURCE_DIRECTORY_PATH10 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/los_tec_txt/"
def get_dtec_from_los_tec_txt_many_sites(source_path, save_path, params):
    list_site_directory_path = get_site_directory_paths_from_directory(source_path)
    for site_directory_path in list_site_directory_path:
        name_site = os.path.basename(site_directory_path)
        save_path_site_1 = get_directory_path(save_path, name_site)
        get_dtec_from_los_tec_txt_1(site_directory_path, save_path_site_1, params)


SAVE_PATH9 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_plots/bor1"
SAVE_PATH11 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_plots/krrs"
SOURCE_DIRECTORY_PATH11 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/krrs"
def plot_graphs_for_site_directory(source_path, save_path):
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_path)
    for path_date_directory in list_of_date_directory_paths:
        name_date_directory = os.path.basename(path_date_directory)
        date = get_date_from_date_directory_name(name_date_directory)
        save_path_date_1 = get_directory_path(save_path, name_date_directory)
        list_od_window_directory_paths = [os.path.join(path_date_directory, directory_name) for directory_name in
                                          os.listdir(path_date_directory)]
        for path_window_directory in list_od_window_directory_paths:
            name_window_directory = os.path.basename(path_window_directory)
            save_path_window_2 = get_directory_path(save_path_date_1, name_window_directory)
            list_od_sat_id_file_paths = get_sat_id_file_paths_from_directory(path_window_directory)
            for sat_id_file_path in list_od_sat_id_file_paths:
                name_sat_id = os.path.basename(sat_id_file_path)[0:3]
                temp_dataframe = read_dtec_file(sat_id_file_path)
                temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                temp_dataframe = add_datetime_column_to_df(temp_dataframe)
                plot_difference_graph(save_path_window_2, name_sat_id, temp_dataframe)
                print(f"ploted {name_date_directory} - {name_window_directory} - {name_sat_id}")







def get_autocorr_dataframe_for_dtec_dataframe(dataframe_dtec, autocorr_limits):
    list_time = list(range(int(dataframe_dtec.iloc[0].loc["timestamp"]), int(dataframe_dtec.iloc[-1].loc["timestamp"]) + 1,
                            PERIOD_CONST))
    dataframe_dtec = dataframe_dtec.set_index("timestamp")
    dupl = dataframe_dtec.index.duplicated()
    if dupl[-1]:
        dataframe_dtec.drop([dataframe_dtec.index[-1]], inplace=True)
    dataframe_dtec = dataframe_dtec.reindex(list_time)
    series_dtec: pd.Series = dataframe_dtec.loc[:, "dtec"]
    bottom_limit = autocorr_limits[0] // PERIOD_CONST * PERIOD_CONST
    list_shift = list(range(bottom_limit, autocorr_limits[1], PERIOD_CONST))
    list_autocorr = []
    for shift in list_shift:
        series_shifted = series_dtec.shift(shift // PERIOD_CONST)
        koef_autocorr = series_dtec.corr(series_shifted)
        list_autocorr.append(koef_autocorr)
    data = {"shift": list_shift, "autocorr": list_autocorr}
    dataframe_autocorr = pd.DataFrame(data)
    dataframe_autocorr = dataframe_autocorr.set_index("shift")
    return dataframe_autocorr


SAVE_PATH12 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/autocorr/krrs/"
def get_autocorr_for_site_directory(source_path, save_path, autocorr_limits):
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_path)
    for path_date_directory in list_of_date_directory_paths:
        name_date_directory = os.path.basename(path_date_directory)
        date = get_date_from_date_directory_name(name_date_directory)
        save_path_date_1 = get_directory_path(save_path, name_date_directory)
        list_od_window_directory_paths = [os.path.join(path_date_directory, directory_name) for directory_name in
                                          os.listdir(path_date_directory)]
        for path_window_directory in list_od_window_directory_paths:
            name_window_directory = os.path.basename(path_window_directory)
            save_path_window_2 = get_directory_path(save_path_date_1, name_window_directory)
            list_od_sat_id_file_paths = get_sat_id_file_paths_from_directory(path_window_directory)
            dataframe_autocorr = pd.DataFrame()
            for sat_id_file_path in list_od_sat_id_file_paths:
                name_sat_id = os.path.basename(sat_id_file_path)[0:3]
                temp_dataframe = read_dtec_file(sat_id_file_path)
                temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                temp_dataframe = temp_dataframe.drop_duplicates()
                dataframe_sat_autocorr = get_autocorr_dataframe_for_dtec_dataframe(temp_dataframe, autocorr_limits)
                dataframe_sat_autocorr = dataframe_sat_autocorr.rename(columns={"shift": "shift",
                                                                                "autocorr": name_sat_id})
                dataframe_autocorr = pd.concat([dataframe_autocorr, dataframe_sat_autocorr], axis=1)
            save_path_autocorr_file = os.path.join(save_path_window_2, "autocorr.txt")
            with open(save_path_autocorr_file, "w") as file_autocorr:
                list_column_names = list(dataframe_autocorr.columns)
                first_line = ""
                for name_column in list_column_names:
                    first_line += f"{name_column:6}\t"
                file_autocorr.write(first_line + "\n")
                for index in dataframe_autocorr.index:
                    list_values = list(dataframe_autocorr.loc[index].values)
                    line = f"{index:6}\t"
                    for value in list_values:
                        line += f"{value:<6.3f}\t"
                    file_autocorr.write(line + "\n")


def temp_main1():
    list_los_file_paths = [r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230224.001.h5.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230225.001.h5.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230301.001.h5_0.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230302.001.h5.hdf5"]
    save_path = SAVE_PATH7
    list_gps_sites = SITES1
    for los_file_path in list_los_file_paths:
        save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path)



if __name__ == "__main__":
    plot_graphs_for_site_directory(SOURCE_DIRECTORY_PATH11, SAVE_PATH11)