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


SOURCE_FILE = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G04.txt"
SOURCE_FILE2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G02.txt"
SOURCE_FILE2_1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/268-2020-09-24/G02.txt"
SOURCE_FILE3 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G08.txt"
SOURCE_FILE16 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G16.txt"
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
                                      "azm": arr_azm, "elm": arr_elm, "gdlat": arr_gdlat, "gdlon": arr_gdlon,})
    return outcome_dataframe


def add_timestamp_column_to_df(dataframe: pd.DataFrame, date):
    new_dataframe = dataframe.assign(timestamp=dataframe["hour"] * 3600 + dataframe["min"] * 60 + dataframe["sec"] +
                                     date.timestamp())
    new_dataframe.loc[:, "datetime"] = pd.to_datetime(new_dataframe.loc[:, "timestamp"], unit="s")
    return new_dataframe


# def create_undivided_dataframe(dataframe: pd.DataFrame):
#     start_time


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


def add_savgov_data(dataframe: pd.DataFrame, window_length=3600, polyorder=2):
    savgov_data = savgol_filter(dataframe.loc[:, "los_tec"], (window_length // PERIOD_CONST + 1), polyorder)
    diff_data = dataframe.loc[:, "los_tec"] - savgov_data
    temp_window_length = window_length // PERIOD_CONST // 2
    new_dataframe = dataframe.assign(savgov=savgov_data,
                                     diff=diff_data.iloc[temp_window_length + 1:-temp_window_length])
    return new_dataframe

#
# def calculate_savgov_diff_data(dataframe: pd.DataFrame, params: Dict):
#     savgov_data = savgol_filter(dataframe.loc[:, "los_tec"], **params)
#     diff_data = dataframe.loc[:, "los_tec"] - savgov_data
#     temp_window_length = (params["window_length"] - 1) // 2
#     new_dataframe = dataframe.assign(savgol=savgov_data,
#                                      diff=diff_data.iloc[temp_window_length:-temp_window_length])
#     return new_dataframe


def fill_small_gaps(dataframe: pd.DataFrame):
    time_series = dataframe.loc[:, "timestamp"]
    diff_time_series = time_series.diff()
    breaking_index_array = diff_time_series.loc[diff_time_series.loc[:] == 2 * PERIOD_CONST].index
    for index in breaking_index_array:
        new_line: pd.Series = dataframe.loc[index-1:index].mean(0)
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


def add_savgov_data_complicate(dataframe: pd.DataFrame, params: Dict, filling=False):
    if filling:
        dataframe = fill_small_gaps(dataframe)
    time_periods = get_time_periods(dataframe.loc[:, "timestamp"], (params["window_length"]))
    new_dataframe = pd.DataFrame()
    for period in time_periods:
        temp_dataframe = dataframe.loc[period[0]:period[1]]
        new_temp_dataframe = add_savgov_data(temp_dataframe, **params)
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe])
    return new_dataframe

def get_ready_dataframe(file, params: Dict, date=dt.datetime.min, filling=False):
    first_dataframe = read_sat_file(file)
    second_dataframe = add_timestamp_column_to_df(first_dataframe, date)
    third_dataframe = add_savgov_data_complicate(second_dataframe, params, filling)
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
#UNREADY
#UNREADY
#
def plot_difference_graph(save_path, name, dataframe, title=None):
    if not title:
        title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    line4, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["diff"], linestyle="-", marker=" ",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_ylim(-1.1, 1.1)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)

# DON"T WORK
def main2():
    params_dicts_list = [
        {"window_length": 61, "polyorder": 2},
        {"window_length": 61, "polyorder": 3},
        {"window_length": 121, "polyorder": 2},
        {"window_length": 121, "polyorder": 3},
        {"window_length": 241, "polyorder": 2},
        {"window_length": 241, "polyorder": 3},
        {"window_length": 61, "polyorder": 4},
        {"window_length": 121, "polyorder": 4},
        {"window_length": 241, "polyorder": 4},
    ]
    names_list = [
        "BASE_267_2020-09-23_G04_WINDOWLENGTH1800_POLYORDER2",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH1800_POLYORDER3",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH3600_POLYORDER2",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH3600_POLYORDER3",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH7200_POLYORDER2",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH7200_POLYORDER3",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH1800_POLYORDER4",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH3600_POLYORDER4",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH7200_POLYORDER4",
    ]
    for index in range(len(params_dicts_list)):
        dataframe = get_ready_dataframe(SOURCE_FILE, params_dicts_list[index])
        plot(SAVE_PATH, names_list[index], dataframe)



def main1():
    dataframe = add_timestamp_column_to_df(read_sat_file(SOURCE_FILE2))
    new_data = dataframe.loc[:, "timestamp"] - dataframe.loc[:, "timestamp"].shift(1)
    non_data: pd.Series = new_data.loc[new_data > 30]
    print(non_data.values)
    print(len(non_data))


def main3():
    params_dicts_list = [
        {"window_length": 3600, "polyorder": 2},
        {"window_length": 7200, "polyorder": 2},
    ]
    names_list = [
        "BASE_267_2020-09-23_G04_WINDOWLENGTH3600_POLYORDER2_DIFF",
        "BASE_267_2020-09-23_G04_WINDOWLENGTH7200_POLYORDER2_DIFF",
    ]
    for index in range(len(params_dicts_list)):
        dataframe = get_ready_dataframe(SOURCE_FILE, params_dicts_list[index])
        title = f"Difference for {names_list[index][:19]} {names_list[index][20:23]} with window " \
                f"{(params_dicts_list[index]['window_length'] - 1) * 30} sec, 2 order"
        plot_difference_graph(SAVE_PATH2, names_list[index], dataframe, title)


def main4():
    params_dicts_list = [
        {"window_length": 3600, "polyorder": 2},
        {"window_length": 7200, "polyorder": 2},
    ]
    names_list = [
        "BASE_267_2020-09-23_G16_WINDOWLENGTH3600_POLYORDER2_DIFF",
        "BASE_267_2020-09-23_G16_WINDOWLENGTH7200_POLYORDER2_DIFF",
    ]
    for index in range(len(params_dicts_list)):
        dataframe = get_ready_dataframe(SOURCE_FILE16, params_dicts_list[index])
        title = f"Difference for {names_list[index][:19]} {names_list[index][20:23]} with window " \
                f"{(params_dicts_list[index]['window_length'] - 1) * 30} sec, 2 order"
        plot_difference_graph(SAVE_PATH2, names_list[index], dataframe, title)

# dataframes contains los_tec and timestamp column
# change los_tec of second dataframe for fitting with first if they have continues observes
# time distance between last two rows must be 30 seconds
def concat_two_dataframes(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame):
    if len(dataframe1) > 0 and len(dataframe2) > 0:
        time_distance = dataframe2.iloc[0].loc["timestamp"] - dataframe1.iloc[-1].loc["timestamp"]
        if time_distance == PERIOD_CONST or time_distance == 2 * PERIOD_CONST:
            last_los_tec_dataframe1 = dataframe1.iloc[-1].loc["los_tec"]
            los_tec_changing = time_distance // PERIOD_CONST * (last_los_tec_dataframe1 - dataframe1.iloc[-2].loc["los_tec"])
            expected_los_tec = last_los_tec_dataframe1 + los_tec_changing
            diff_between_factual_and_expected = dataframe2.iloc[0].loc["los_tec"] - expected_los_tec
            dataframe2.loc[:, "los_tec"] = dataframe2.loc[:, "los_tec"] - diff_between_factual_and_expected
        outcome_dataframe: pd.DataFrame = pd.concat([dataframe1, dataframe2])
    elif len(dataframe1) > 0 and len(dataframe2) == 0:
        outcome_dataframe = dataframe1
    else:
        outcome_dataframe = dataframe2
    return outcome_dataframe




# input_list is list of tuples (date_for_the_file: datetime, path_to_file)
# list must be sorted by the date
def get_ready_dataframe_from_list_for_one_satellite(input_list: List, params: Dict):
    outcome_dataframe = pd.DataFrame()
    for date, path_to_file in input_list:
        temp_dataframe = get_dataframe_with_timestamp(path_to_file, date=date)
        outcome_dataframe = concat_two_dataframes(outcome_dataframe, temp_dataframe)
    outcome_dataframe.sort_values(by="timestamp", inplace=True)
    outcome_dataframe.reset_index(inplace=True, drop=True)
    outcome_dataframe = add_savgov_data_complicate(outcome_dataframe, params=params, filling=True)
    return outcome_dataframe


def main5():
    list_of_files = [(dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc), SOURCE_FILE2),
                     (dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc), SOURCE_FILE2_1)]
    dataframe = get_ready_dataframe_from_list_for_one_satellite(list_of_files, {"window_length": 3600})
    dataframe.loc[:, "datetime"] = pd.to_datetime(dataframe.loc[:, "timestamp"], unit="s")
    plot(SAVE_PATH5, "draph", dataframe)
    dataframe.to_csv(os.path.join(SAVE_PATH5, "dataframe.csv"))


def main6():
    name1 = "BASE_267_2020-09-23_G02_WINDOWLENGTH3600_POLYORDER2_DIFF"
    dataframe = get_ready_dataframe(SOURCE_FILE2, {"window_length": 3600, "polyorder": 2},
                                    date=dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc))
    title = f"Difference for {name1[:19]} {name1[20:23]} with window " \
            f"{3600} sec, 2 order"
    plot_difference_graph(SAVE_PATH5, name1, dataframe, title)
    dataframe.to_csv(os.path.join(SAVE_PATH5, name1))
    name2 = "BASE_268_2020-09-24_G02_WINDOWLENGTH3600_POLYORDER2_DIFF"
    dataframe = get_ready_dataframe(SOURCE_FILE2_1, {"window_length": 3600, "polyorder": 2},
                                    date=dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc))
    title = f"Difference for {name2[:19]} {name2[20:23]} with window " \
            f"{3600} sec, 2 order"
    plot_difference_graph(SAVE_PATH5, name2, dataframe, title)
    dataframe.to_csv(os.path.join(SAVE_PATH5, name2))


def main_test_correct():
    list_of_params = [{"window_length": 3600}, {"window_length": 7200}]
    directory_name1 = os.path.basename(os.path.dirname(SOURCE_DIRECTORY1))
    directory_name2 = os.path.basename(os.path.dirname(SOURCE_DIRECTORY2))
    date1 = dt.datetime(year=int(directory_name1[4:8]),
                        month=int(directory_name1[9:11]),
                        day=int(directory_name1[12:14]), tzinfo=dt.timezone.utc)
    date2 = dt.datetime(year=int(directory_name2[4:8]),
                        month=int(directory_name2[9:11]),
                        day=int(directory_name2[12:14]), tzinfo=dt.timezone.utc)
    for params in list_of_params:
        print(1, params)
        list_of_files = os.listdir(SOURCE_DIRECTORY1)
        for file in list_of_files:
            print("\t", file)
            path1 = os.path.join(SOURCE_DIRECTORY1, file)
            path2 = os.path.join(SOURCE_DIRECTORY2, file)
            dataframe1: pd.DataFrame = get_ready_dataframe(path2, params, date1, True)
            dataframe2: pd.DataFrame = get_ready_dataframe_from_list_for_one_satellite([(date1, path1),
                                                                                        (date2, path2)], params)
            dataframe1.set_index("timestamp", inplace=True)
            dataframe2.set_index("timestamp", inplace=True)
            dataframe1 = dataframe1.loc[dataframe1.loc[:, "diff"].notna()]
            temp_series1 = dataframe1.loc[:, "diff"]
            temp_series2 = dataframe2.loc[dataframe1.index, "diff"]
            temp_diff_series: pd.Series = temp_series1 - temp_series2
            print(f"\tmin - {temp_diff_series.min():<10}, max - {temp_diff_series.max():<10} for {file[0:3]}")
# KeyError: "None of [Index([1600822869.0, 1600822899.0, 1600822929.0, 1600822959.0, 1600822989.0,\n       1600823019.0, 1600823049.0, 1600823079.0, 1600823109.0, 1600823139.0,\n       ...\n       1600869189.0, 1600869219.0, 1600869249.0, 1600869279.0, 1600869309.0,\n       1600869339.0, 1600869369.0, 1600869399.0, 1600869429.0, 1600869459.0],\n      dtype='object', name='timestamp', length=858)] are in the [index]"


def save_diff_dataframe(dataframe: pd.DataFrame, path):
    pass


SOURCE_DIRECTORY3 = r""
def main8():
    list_of_params = [{"window_length": 3600}, {"window_length": 7200}]
    for directory in [SOURCE_DIRECTORY1, SOURCE_DIRECTORY2]:
        print(1, directory)
        directory_name = os.path.basename(os.path.dirname(directory))
        save_directory_path = os.path.join(SAVE_PATH6, directory_name)
        if not os.path.exists(save_directory_path):
            os.mkdir(save_directory_path)
        date = dt.datetime(year=int(directory_name[4:8]),
                           month=int(directory_name[9:11]),
                           day=int(directory_name[12:14]), tzinfo=dt.timezone.utc)
        list_of_files = os.listdir(directory)
        for params in list_of_params:
            print(2, params)
            save_params_path = os.path.join(save_directory_path, f"Window_{params['window_length']}_Seconds")
            if not os.path.exists(save_params_path):
                os.mkdir(save_params_path)
            for file in list_of_files:
                print(3, file)
                path = os.path.join(directory, file)
                sat_id = int(file[1:3])
                dataframe: pd.DataFrame = get_ready_dataframe(path, params, date, True)
                mask = dataframe.loc[:, "diff"].notna()
                dataframe = dataframe.loc[mask]
                new_file = os.path.join(save_params_path, f"G{sat_id:0=2}.txt")
                with open(new_file, "w") as write_file:
                    write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dTEC':<6}\t{'azm':<6}\t{'elm':<6}"
                                     f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
                    for index in dataframe.index:
                        hour = dataframe.loc[index, "hour"]
                        min = dataframe.loc[index, "min"]
                        sec = dataframe.loc[index, "sec"]
                        dtec = dataframe.loc[index, "diff"]
                        azm = dataframe.loc[index, "azm"]
                        elm = dataframe.loc[index, "elm"]
                        gdlat = dataframe.loc[index, "gdlat"]
                        gdlon = dataframe.loc[index, "gdlon"]
                        write_file.write(f"{hour:<4}\t{min:<4}\t{sec:<4.0f}\t{dtec:<6.3f}\t{azm:<6.2f}\t{elm:<6.2f}"
                                         f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")





if __name__ == "__main__":
    main8()