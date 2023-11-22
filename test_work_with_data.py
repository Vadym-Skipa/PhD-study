import h5py
import pandas as pd
import numpy as np
import reader_cmn as rcmn
import read_los_file as rlos
import datetime as dt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import matplotlib.figure as fig
import os
import re
import multiprocessing as mp

SAVE_TEMP_FILES = r"/home/vadymskipa/Documents/Temporary_files"
TEST_LOS_FILE = r"/home/vadymskipa/Downloads/los_20220604.001.h5"
TEST_SITE_FILE = r"/home/vadymskipa/Downloads/site_20220604.001.h5"
TEST_SITE = "bogi"
TEST_CMN = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/CMN_data_from_program/bogi155-2022-06-04.Cmn"
TEST_MAX_TIME_CADENCE = 5.0 / 60
TEST_MAX_TIME_CADENCE_UNIX = TEST_MAX_TIME_CADENCE * 3600
TEST_CMN_FILES = (r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/CMN_data_from_program/bogi155-2022-06-04.Cmn",
                  )
TEST_DATE = dt.datetime(year=2022, month=6, day=4, tzinfo=dt.timezone.utc)
TEST_SAVE_DIRECTORY_FOR_DIFFERENCE = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files"
TEST_CMN_FILES_20220604 = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/CMN_files_20220604"


def test_split_date_by_time_cmn(prn=0):
    data_cmn = rcmn.read_cmn_file_short_pd(TEST_CMN)
    array_of_prns = np.sort(np.unique(data_cmn["prn"]))
    prn0 = array_of_prns[prn]
    data_for_prn0 = data_cmn[data_cmn["prn"] == prn0].reset_index(drop=True)
    shift_time_array = data_for_prn0["time"].shift(1)
    diff_time_array = data_for_prn0["time"] - shift_time_array
    breaking_points = diff_time_array[diff_time_array > TEST_MAX_TIME_CADENCE]
    array_of_breaking_points = breaking_points.index
    # start_point = 0
    # for point in array_of_breaking_points:
    #     print(data_for_prn0[start_point:point])
    #     start_point = point
    # print(data_for_prn0[start_point:])
    return [array_of_breaking_points, data_for_prn0]


def test_split_date_by_time_los_file(prn=0):
    data_los = rlos.get_data_for_one_site_GPS_pd(TEST_LOS_FILE, TEST_SITE)
    array_of_prns = np.sort(np.unique(data_los["sat_id"]))
    prn0 = array_of_prns[prn]
    data_for_prn0 = pd.DataFrame(data_los[data_los["sat_id"] == prn0]).reset_index(drop=True)
    time_array = data_for_prn0.loc[:, ["ut1_unix", "ut2_unix"]].mean(axis=1)
    temp = dt.datetime(year=2022, month=6, day=4, tzinfo=dt.timezone.utc).timestamp()
    new_time = (time_array - temp) / 3600.0
    data_for_prn0 = data_for_prn0.assign(time=new_time)
    shift_time_array = time_array.shift(1)
    diff_time_array = time_array - shift_time_array
    breaking_points = diff_time_array[diff_time_array > TEST_MAX_TIME_CADENCE_UNIX]
    array_of_breaking_points = breaking_points.index
    # start_point = 0
    # for point in array_of_breaking_points:
    #     print(data_for_prn0[start_point:point])
    #     start_point = point
    # print(data_for_prn0[start_point:])
    return [array_of_breaking_points, data_for_prn0]


def _create_diff_between_two_function_in_limits(function1, function2, start, end, step,
                                                x_axis_name="time", y_axis_name="stec"):
    x_values = np.arange(start, end, step)
    y_values = [function1(i) - function2(i) for i in x_values]
    result = pd.DataFrame({x_axis_name: np.array(x_values, dtype="f"), y_axis_name: np.array(y_values, dtype="f")})
    return result


def test_calculate_diff_between_madrigal_and_cmn():
    array_of_breaking_points_cmn, data_for_prn0_cmn = test_split_date_by_time_cmn()
    array_of_breaking_points_los, data_for_prn0_los = test_split_date_by_time_los_file()
    array_of_time_points = []
    array_of_time_points.append((data_for_prn0_cmn.iloc[0].loc["time"], "start"))
    array_of_time_points.append((data_for_prn0_cmn.iloc[-1].loc["time"], "end"))
    array_of_time_points.append((data_for_prn0_los.iloc[0].loc["time"], "start"))
    array_of_time_points.append((data_for_prn0_los.iloc[-1].loc["time"], "end"))
    if not array_of_breaking_points_cmn.empty:
        for el in array_of_breaking_points_cmn:
            array_of_time_points.append((data_for_prn0_cmn.iloc[el - 1].loc["time"], "end"))
            array_of_time_points.append((data_for_prn0_cmn.iloc[el].loc["time"], "start"))
    if not array_of_breaking_points_los.empty:
        for el in array_of_breaking_points_los:
            array_of_time_points.append((data_for_prn0_los.iloc[el - 1].loc["time"], "end"))
            array_of_time_points.append((data_for_prn0_los.iloc[el].loc["time"], "start"))
    array_of_time_points.sort(key=lambda x: x[0])
    previous_point = (0, 0)
    result = None
    function_cmn = interp1d(data_for_prn0_cmn["time"], data_for_prn0_cmn["stec"])
    function_los = interp1d(data_for_prn0_los["time"], data_for_prn0_los["los_tec"])
    for point in array_of_time_points:
        if point[1] == "end" and previous_point[1] == "start":
            temp_result = _create_diff_between_two_function_in_limits(function_los, function_cmn, previous_point[0],
                                                                      point[0], 1.0 / 120)
            result = pd.concat([result, temp_result], ignore_index=True)
        previous_point = point
    first_mean = result["stec"].mean()
    first_std = result["stec"].std()
    new_result = result[abs(result["stec"] - first_mean) < first_std]
    mean = new_result["stec"].mean()

    figure: fig.Figure = plt.figure(layout="tight")
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("STEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    line1, = axes1.plot(data_for_prn0_cmn["time"], data_for_prn0_cmn["stec"], label="Data from GPS-TEC", linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1.1, markersize=1)

    line2, = axes1.plot(data_for_prn0_los["time"], data_for_prn0_los["los_tec"], label="Data from MADRIGAL",
                        linestyle=" ", marker=".", color="red", markeredgewidth=1.1, markersize=1)
    axes1.legend()
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line3, = axes2.plot((result.iloc[0].loc["time"], result.iloc[-1].loc["time"]), (mean, mean), color="red",
                        linewidth=1, linestyle="-", label=f"mean = {mean}", )
    line4, = axes2.plot(result["time"], result["stec"], linestyle=" ", marker=".", markeredgewidth=1, markersize=1,
                        label="Difference between Madrigal data and GPS-TEC")
    axes2.legend()
    plt.savefig(os.path.join(SAVE_TEMP_FILES, "Difference_between_los_and_cmn_bogi20220604_prn1.svg"))
    plt.close(figure)

    # plt.plot((result.iloc[0].loc["time"], result.iloc[-1].loc["time"]), (mean, mean), color="red", linewidth=1, linestyle="-")
    # plt.plot(result["time"], result["stec"], linestyle=" ", marker=".", markeredgewidth=1, markersize=1)
    # plt.savefig(os.path.join(SAVE_TEMP_FILES, "Difference_between_los_and_cmn_bogi20220604_prn1.svg"))
    # plt.close()
    print(mean)


def _split_data_by_time(time_array, time_cadence=TEST_MAX_TIME_CADENCE):
    shift_time_array = time_array.shift(1)
    diff_time_array = time_array - shift_time_array
    breaking_points = diff_time_array[diff_time_array > time_cadence]
    array_of_breaking_indexes = breaking_points.index
    return array_of_breaking_indexes


def _gets_common_time_periods(time_array1, time_array2):
    array_of_breaking_indexes1 = _split_data_by_time(time_array1)
    array_of_breaking_indexes2 = _split_data_by_time(time_array2)

    time_periods1 = []
    previous_point = time_array1.iloc[0]
    for index in array_of_breaking_indexes1:
        time_periods1.append((previous_point, time_array1.iloc[index - 1]))
        previous_point = time_array1.iloc[index]
    time_periods1.append((previous_point, time_array1.iloc[-1]))
    time_periods2 = []
    previous_point = time_array2.iloc[0]
    for index in array_of_breaking_indexes2:
        time_periods2.append((previous_point, time_array2.iloc[index - 1]))
        previous_point = time_array2.iloc[index]
    time_periods2.append((previous_point, time_array2.iloc[-1]))

    time_periods = []
    for start1, end1 in time_periods1:
        for start2, end2 in time_periods2:
            if start2 < start1 < end2:
                if end2 < end1:
                    time_periods.append((start1, end2))
                else:
                    time_periods.append((start1, end1))
            elif start1 < start2 < end1:
                if end2 < end1:
                    time_periods.append((start2, end2))
                else:
                    time_periods.append((start2, end1))

    return time_periods
    # array_of_time_points = []
    # array_of_time_points.append((time_array1.iloc[0], "start"))
    # array_of_time_points.append((time_array1.iloc[-1], "end"))
    # array_of_time_points.append((time_array2.iloc[0], "start"))
    # array_of_time_points.append((time_array2.iloc[-1], "end"))
    # if not array_of_breaking_points1.empty:
    #     for el in array_of_breaking_points1:
    #         array_of_time_points.append((time_array1.iloc[el - 1], "end"))
    #         array_of_time_points.append((time_array1.iloc[el], "start"))
    # if not array_of_breaking_points2.empty:
    #     for el in array_of_breaking_points2:
    #         array_of_time_points.append((time_array2.iloc[el - 1], "end"))
    #         array_of_time_points.append((time_array2.iloc[el], "start"))
    # array_of_time_points.sort(key=lambda x: x[0])
    # previous_point = (0, 0)
    # result_time_periods = []
    # for point in array_of_time_points:
    #     if point[1] == "end" and previous_point[1] == "start":
    #         result_time_periods.append((previous_point[0], point[0]))
    #     previous_point = point
    # return result_time_periods


def _save_plot_of_difference(data_los, data_cmn, data_difference, save_path, title=None, format=None):
    first_mean = data_difference["diff_stec"].mean()
    first_std = data_difference["diff_stec"].std()
    new_result = data_difference[abs(data_difference["diff_stec"] - first_mean) < first_std]
    mean = new_result["diff_stec"].mean()

    figure: fig.Figure = plt.figure(layout="tight")
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("STEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    line1, = axes1.plot(data_cmn["time"], data_cmn["stec"], label="Data from GPS-TEC", linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)

    line2, = axes1.plot(data_los["time"], data_los["los_tec"], label="Data from MADRIGAL",
                        linestyle=" ", marker=".", color="red", markeredgewidth=1, markersize=1.1)
    axes1.legend()
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line3, = axes2.plot((data_difference.iloc[0].loc["time"], data_difference.iloc[-1].loc["time"]), (mean, mean),
                        color="red",
                        linewidth=1, linestyle="-", label=f"Mean2 = {mean}", )
    line4, = axes2.plot(data_difference["time"], data_difference["diff_stec"], linestyle=" ", marker=".", markeredgewidth=1,
                        markersize=1,
                        label="Difference between Madrigal data and GPS-TEC")
    axes2.legend()
    axes2.set_ylim(-3.0, 3.0)
    if title:
        figure.suptitle(title)
    plt.savefig(save_path + f".{format}", dpi=300, format=format)
    plt.close(figure)


def calculate_diff_between_madrigal_and_cmn_for_sites(los_file=TEST_LOS_FILE, array_of_cmn_files=TEST_CMN_FILES,
                                                      date=TEST_DATE,
                                                      save_directory=TEST_SAVE_DIRECTORY_FOR_DIFFERENCE):
    print(f"START___{dt.datetime.now()}")
    for file in array_of_cmn_files:
        site = os.path.basename(file)[0:4]
        data_cmn = rcmn.read_cmn_file_short_pd(file)
        data_los = rlos.get_data_for_one_site_GPS_pd(los_file, site)
        array_of_prns_cmn = np.unique(data_cmn["prn"])
        array_of_prns_los = np.unique(data_los["sat_id"])
        array_of_prns = np.intersect1d(array_of_prns_los, array_of_prns_cmn)
        for prn in array_of_prns:
            data_cmn_for_prn = data_cmn[data_cmn["prn"] == prn].reset_index(drop=True)
            data_los_for_prn = data_los[data_los["sat_id"] == prn].reset_index(drop=True)
            time_array = data_los_for_prn.loc[:, ["ut1_unix", "ut2_unix"]].mean(axis=1)
            temp = date.timestamp()
            new_time = (time_array - temp) / 3600.0
            data_los_for_prn = data_los_for_prn.assign(time=new_time)
            time_periods = _gets_common_time_periods(data_cmn_for_prn["time"], data_los_for_prn["time"])
            difference = None
            function_cmn = interp1d(data_cmn_for_prn["time"], data_cmn_for_prn["stec"], kind="cubic")
            function_los = interp1d(data_los_for_prn["time"], data_los_for_prn["los_tec"], kind="cubic")
            print(time_periods)
            for start, end in time_periods:
                temp_result = _create_diff_between_two_function_in_limits(function_los, function_cmn, start + 1.0 / 300,
                                                                          end - 1.0 / 300, 1.0 / 120)
                difference = pd.concat([difference, temp_result], ignore_index=True)
            filename = f"{site}_prn{prn:0=2}_{date.year}{date.month:0=2}{date.day:0=2}_diff_los_cmn.png"
            save_path = os.path.join(save_directory, filename)
            _save_plot_of_difference(data_los_for_prn, data_cmn_for_prn, difference, save_path)
            print(f"___{prn:0=2}___{dt.datetime.now()}")


#
#
#
#
#


def _filter_cmn_files_by_availability_in_site_file(cmn_files, site_file_path):
    with h5py.File(site_file_path, "r") as file:
        site_arr = file["Data"]["Table Layout"]["gps_site"]
        sites_from_file = np.unique(site_arr)
    result_list = [file for file in cmn_files if np.array([file], dtype="S4")[0] in sites_from_file]
    return result_list


def _get_all_cmn_file_names_in_directory(directory):
    list_of_objects_in_directory = os.listdir(directory)
    result_list = [el for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory, el)) and re.search("(\.cmn)$", el.lower()))]
    return result_list


def _filter_cmn_files_by_date(cmn_files, date: dt.date):
    result_list = [cmn_file for cmn_file in cmn_files if cmn_file[8:18] == date.isoformat()]
    return result_list


def get_cmn_files_from_directory(directory, site_file_path=None, date=None):
    list_of_filenames = _get_all_cmn_file_names_in_directory(directory)
    if site_file_path:
        list_of_filenames = _filter_cmn_files_by_availability_in_site_file(list_of_filenames, site_file_path)
    if date:
        list_of_filenames = _filter_cmn_files_by_date(list_of_filenames, date.date())
    return list_of_filenames


def _calculate_difference_between_madrigal_and_cmn_for_site_and_prn(data_los_for_site_and_prn, data_cmn_for_prn):
    time_periods = _gets_common_time_periods(data_cmn_for_prn.iloc[:, 0], data_los_for_site_and_prn.iloc[:, 0])
    difference = None
    function_cmn = interp1d(data_cmn_for_prn.iloc[:, 0], data_cmn_for_prn.iloc[:, 1])
    function_los = interp1d(data_los_for_site_and_prn.iloc[:, 0], data_los_for_site_and_prn.iloc[:, 1])
    for start, end in time_periods:
        temp_result = _create_diff_between_two_function_in_limits(function_los, function_cmn, start + 1.0 / 3600,
                                                                  end - 1.0 / 3600, 1.0 / 120, "time", "diff_stec")
        difference = pd.concat([difference, temp_result], ignore_index=True)
    return difference


def calculate_difference_between_madrigal_and_cmn_for_site(los_file_path, cmn_file_path, date=None):
    data_cmn = rcmn.read_cmn_file_short_pd(cmn_file_path)
    site = os.path.basename(cmn_file_path)[0:4]
    data_los = rlos.get_data_for_one_site_GPS_pd(los_file_path, site)
    time_array_los = data_los.loc[:, ["ut1_unix", "ut2_unix"]].mean(axis=1)
    if not date:
        date = dt.datetime.strptime(os.path.basename(los_file_path)[4:12], r"%Y%m%d")
    temp = date.timestamp()
    new_time_array_los = (time_array_los - temp) / 3600.0
    data_los = data_los.assign(time=new_time_array_los)
    array_of_prns_cmn = np.unique(data_cmn["prn"])
    array_of_prns_los = np.unique(data_los["sat_id"])
    array_of_prns = np.intersect1d(array_of_prns_los, array_of_prns_cmn)
    difference = None
    for prn in array_of_prns:
        data_cmn_prn = (data_cmn[data_cmn["prn"] == prn].reset_index(drop=True)).loc[:, ["time", "stec"]]
        date_los_prn = (data_los[data_los["sat_id"] == prn].reset_index(drop=True).loc[:, ["time", "los_tec"]])
        difference_prn = _calculate_difference_between_madrigal_and_cmn_for_site_and_prn(date_los_prn, data_cmn_prn)
        difference_prn = difference_prn.assign(prn=np.full(len(difference_prn), prn, dtype="i"))
        difference = pd.concat([difference, difference_prn], ignore_index=True)
    cmn = data_cmn.loc[:,["time", "stec", "prn"]]
    los = data_los.loc[:,["time", "los_tec", "sat_id"]]
    return (difference, los, cmn)


def _get_datetime_from_los_file(los_file_path):
    year = int(os.path.basename(los_file_path)[4:8])
    month = int(os.path.basename(los_file_path)[8:10])
    day = int(os.path.basename(los_file_path)[10:12])
    date = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
    return date


def _save_difference(difference: pd.DataFrame, save_path, chunks=None, maxshape=None):
    with h5py.File(save_path, "w") as file:
        grp1 = file.create_group("Data")
        grp2 = file.create_group("Metadata")
        temp_type = np.dtype([("time", "f"), ("diff_stec", "f"), ("prn", "i"), ("site", "S4")])
        difference.astype({"time": "f", "diff_stec": "f", "prn": "i", "site": "S4"})
        dst = grp1.create_dataset("Table Layout", shape=(len(difference), ), dtype=temp_type, chunks=chunks,
                                  maxshape=maxshape)
        dst["time"] = difference["time"].to_numpy()
        dst["diff_stec"] = difference["diff_stec"]
        dst["prn"] = difference["prn"]
        dst["site"] = difference["site"]


def _add_difference(difference: pd.DataFrame, save_path):
    if os.path.exists(save_path):
        with h5py.File(save_path, "r+") as file:
            dst: h5py.Dataset = file["Data"]["Table Layout"]
            dst.resize(dst.shape[0] + difference.shape[0], axis=0)
            difference.astype({"time": "f", "diff_stec": "f", "prn": "i", "site": "S4"})
            dst[-difference.shape[0]:, "time"] = difference["time"].to_numpy()
            dst[-difference.shape[0]:, "diff_stec"] = difference["diff_stec"].to_numpy()
            dst[-difference.shape[0]:, "prn"] = difference["prn"].to_numpy()
            dst[-difference.shape[0]:, "site"] = difference["site"].to_numpy()
    else:
        _save_difference(difference, save_path, chunks=True, maxshape=(None, ))



TEST_SAVE_DIRECTORY_FOR_MAIN1 = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/" \
                                r"Difference_between_LOS_and_CMN_files/2022-06-04"

def _get_calculated_sites(save_directory=TEST_SAVE_DIRECTORY_FOR_MAIN1):
    result = []
    if os.path.exists(os.path.join(save_directory, "difference.h5")):
        with h5py.File(os.path.join(save_directory, "difference.h5")) as file:
            sites = np.unique(file["Data"]["Table Layout"]["site"])
        sites = [site.decode("ascii") for site in sites]
        result = sites
    return result


def main1(los_file_path=TEST_LOS_FILE, directory=TEST_CMN_FILES_20220604,
          save_directory=TEST_SAVE_DIRECTORY_FOR_MAIN1, site_file_path=TEST_SITE_FILE):
    date = _get_datetime_from_los_file(los_file_path)
    start = dt.datetime.now()
    print(f"Start------------{start}")
    list_of_cmn_filenames = get_cmn_files_from_directory(directory, site_file_path, date)
    list_of_exceptions = _get_calculated_sites(save_directory)
    list_of_cmn_filenames = [file for file in list_of_cmn_filenames if file[0:4] not in list_of_exceptions]
    difference = None
    max = len(list_of_cmn_filenames)
    i = 0
    for cmn_filename in list_of_cmn_filenames:
        print(f"___Calculating___{dt.datetime.now() - start}")
        diff, los, cmn = calculate_difference_between_madrigal_and_cmn_for_site(los_file_path,
                                                                                os.path.join(directory, cmn_filename),
                                                                                date)
        site = cmn_filename[0:4]
        diff = diff.assign(site=np.full(len(diff), site, dtype="S4"))
        # difference = pd.concat([difference, diff], ignore_index=True)
        temp_save_directory = os.path.join(save_directory, site)
        if not os.path.isdir(temp_save_directory):
            os.mkdir(temp_save_directory)
        print(f"___Saving________{dt.datetime.now() - start}")
        for prn in np.unique(diff["prn"]):
            diff_prn = diff[diff["prn"] == prn].loc[:,["time", "diff_stec"]]
            los_prn = los[los["sat_id"] == prn].loc[:,["time", "los_tec"]]
            cmn_prn = cmn[cmn["prn"] == prn].loc[:,["time", "stec"]]
            savename = f"{site}_prn{prn:0=2}_{date.year}{date.month:0=2}{date.day:0=2}_diff_los_cmn"
            if not os.path.isdir(os.path.join(temp_save_directory, "svg")):
                os.mkdir(os.path.join(temp_save_directory, "svg"))
            if not os.path.isdir(os.path.join(temp_save_directory, "png")):
                os.mkdir(os.path.join(temp_save_directory, "png"))
            _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "svg", savename),
                                     f"{site}_prn{prn:0=2}", format="svg")
            _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "png", savename),
                                     f"{site}_prn{prn:0=2}", format="png")
        _save_difference(diff, os.path.join(temp_save_directory, site + "_diff.h5"))
        _add_difference(diff, os.path.join(save_directory, "difference.h5"))
        i += 1
        print(f"{i:0=3}/{max}---{site}---{dt.datetime.now() - start}")


# try 2
def _subprocess_main2(diff, los, cmn, site, save_directory, date, start, i, max):
    diff = diff.assign(site=np.full(len(diff), site, dtype="S4"))
    # difference = pd.concat([difference, diff], ignore_index=True)
    temp_save_directory = os.path.join(save_directory, site)
    if not os.path.isdir(temp_save_directory):
        os.mkdir(temp_save_directory)
    print(f"___Saving__{site}__{dt.datetime.now() - start}")
    for prn in np.unique(diff["prn"]):
        diff_prn = diff[diff["prn"] == prn].loc[:, ["time", "diff_stec"]]
        los_prn = los[los["sat_id"] == prn].loc[:, ["time", "los_tec"]]
        cmn_prn = cmn[cmn["prn"] == prn].loc[:, ["time", "stec"]]
        savename = f"{site}_prn{prn:0=2}_{date.year}{date.month:0=2}{date.day:0=2}_diff_los_cmn"
        if not os.path.isdir(os.path.join(temp_save_directory, "svg")):
            os.mkdir(os.path.join(temp_save_directory, "svg"))
        if not os.path.isdir(os.path.join(temp_save_directory, "png")):
            os.mkdir(os.path.join(temp_save_directory, "png"))
        _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "svg", savename),
                                 f"{site}_prn{prn:0=2}", format="svg")
        _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "png", savename),
                                 f"{site}_prn{prn:0=2}", format="png")
    _save_difference(diff, os.path.join(temp_save_directory, site + "_diff.h5"))
    _add_difference(diff, os.path.join(save_directory, "difference.h5"))
    print(f"{i:0=3}/{max}---{site}---{dt.datetime.now() - start}---{dt.datetime.now().time()}")


def main2(los_file_path=TEST_LOS_FILE, directory=TEST_CMN_FILES_20220604,
          save_directory=TEST_SAVE_DIRECTORY_FOR_MAIN1, site_file_path=TEST_SITE_FILE):
    date = _get_datetime_from_los_file(los_file_path)
    start = dt.datetime.now()
    print(f"Start------------{start.time()}")
    list_of_cmn_filenames = get_cmn_files_from_directory(directory, site_file_path, date)
    list_of_exceptions = _get_calculated_sites(save_directory)
    list_of_cmn_filenames = [file for file in list_of_cmn_filenames if file[0:4] not in list_of_exceptions]
    difference = None
    max = len(list_of_cmn_filenames)
    i = 1
    for cmn_filename in list_of_cmn_filenames:
        print(f"___Calculating___{dt.datetime.now() - start}")
        diff, los, cmn = calculate_difference_between_madrigal_and_cmn_for_site(los_file_path,
                                                                                os.path.join(directory, cmn_filename),
                                                                                date)
        site = cmn_filename[0:4]
        saving = mp.Process(target=_subprocess_main2, args=(diff, los, cmn, site, save_directory, date, start, i, max))
        saving.start()
        i += 1


# try 3
def calculate_difference_between_madrigal_and_cmn_for_site3(los_file_path, cmn_file_path, site_array_los, date=None):
    data_cmn = rcmn.read_cmn_file_short_pd(cmn_file_path)
    site = os.path.basename(cmn_file_path)[0:4]
    site_indicies = site_array_los == site.encode("ascii")
    data_los = rlos.get_data_by_indecies_GPS_pd(los_file_path, site_indicies)
    time_array_los = data_los.loc[:, ["ut1_unix", "ut2_unix"]].mean(axis=1)
    if not date:
        date = dt.datetime.strptime(os.path.basename(los_file_path)[4:12], r"%Y%m%d")
    temp = date.timestamp()
    new_time_array_los = (time_array_los - temp) / 3600.0
    data_los = data_los.assign(time=new_time_array_los)
    array_of_prns_cmn = np.unique(data_cmn["prn"])
    array_of_prns_los = np.unique(data_los["sat_id"])
    array_of_prns = np.intersect1d(array_of_prns_los, array_of_prns_cmn)
    difference = None
    for prn in array_of_prns:
        data_cmn_prn = (data_cmn[data_cmn["prn"] == prn].reset_index(drop=True)).loc[:, ["time", "stec"]]
        date_los_prn = (data_los[data_los["sat_id"] == prn].reset_index(drop=True).loc[:, ["time", "los_tec"]])
        difference_prn = _calculate_difference_between_madrigal_and_cmn_for_site_and_prn(date_los_prn, data_cmn_prn)
        difference_prn = difference_prn.assign(prn=np.full(len(difference_prn), prn, dtype="i"))
        difference = pd.concat([difference, difference_prn], ignore_index=True)
    cmn = data_cmn.loc[:,["time", "stec", "prn"]]
    los = data_los.loc[:,["time", "los_tec", "sat_id"]]
    return (difference, los, cmn)


def main3(los_file_path=TEST_LOS_FILE, directory=TEST_CMN_FILES_20220604,
          save_directory=TEST_SAVE_DIRECTORY_FOR_MAIN1, site_file_path=TEST_SITE_FILE):
    date = _get_datetime_from_los_file(los_file_path)
    start = dt.datetime.now()
    print(f"Start------------{start}")
    list_of_cmn_filenames = get_cmn_files_from_directory(directory, site_file_path, date)
    list_of_exceptions = _get_calculated_sites(save_directory)
    list_of_cmn_filenames = [file for file in list_of_cmn_filenames if file[0:4] not in list_of_exceptions]
    difference = None
    max = len(list_of_cmn_filenames)
    i = 0
    with h5py.File(los_file_path, "r") as file:
        sites = file["Data"]["Table Layout"]["gps_site"]
    for cmn_filename in list_of_cmn_filenames:
        print(f"___Calculating___{dt.datetime.now() - start}")
        diff, los, cmn = calculate_difference_between_madrigal_and_cmn_for_site3(los_file_path,
                                                                                os.path.join(directory, cmn_filename),
                                                                                sites, date)
        site = cmn_filename[0:4]
        diff = diff.assign(site=np.full(len(diff), site, dtype="S4"))
        # difference = pd.concat([difference, diff], ignore_index=True)
        temp_save_directory = os.path.join(save_directory, site)
        if not os.path.isdir(temp_save_directory):
            os.mkdir(temp_save_directory)
        print(f"___Saving________{dt.datetime.now() - start}")
        for prn in np.unique(diff["prn"]):
            diff_prn = diff[diff["prn"] == prn].loc[:,["time", "diff_stec"]]
            los_prn = los[los["sat_id"] == prn].loc[:,["time", "los_tec"]]
            cmn_prn = cmn[cmn["prn"] == prn].loc[:,["time", "stec"]]
            savename = f"{site}_prn{prn:0=2}_{date.year}{date.month:0=2}{date.day:0=2}_diff_los_cmn"
            if not os.path.isdir(os.path.join(temp_save_directory, "svg")):
                os.mkdir(os.path.join(temp_save_directory, "svg"))
            if not os.path.isdir(os.path.join(temp_save_directory, "png")):
                os.mkdir(os.path.join(temp_save_directory, "png"))
            _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "svg", savename),
                                     f"{site}_prn{prn:0=2}", format="svg")
            _save_plot_of_difference(los_prn, cmn_prn, diff_prn, os.path.join(temp_save_directory, "png", savename),
                                     f"{site}_prn{prn:0=2}", format="png")
        _save_difference(diff, os.path.join(temp_save_directory, site + "_diff.h5"))
        _add_difference(diff, os.path.join(save_directory, "difference.h5"))
        i += 1
        print(f"{i:0=3}/{max}---{site}---{dt.datetime.now() - start}")




def test_main():
    los_file_path = TEST_LOS_FILE
    directory = TEST_CMN_FILES_20220604
    save_directory = r"/home/vadymskipa/Documents/Temporary_files"
    site_file_path = TEST_SITE_FILE
    date = _get_datetime_from_los_file(los_file_path)
    list_of_cmn_filenames = get_cmn_files_from_directory(directory, site_file_path, date)
    for cmn_filename in list_of_cmn_filenames:
        if "bogi" in cmn_filename:
            data = calculate_difference_between_madrigal_and_cmn_for_site(los_file_path,
                                                                          os.path.join(directory, cmn_filename), date)
            prns = np.unique(data[0]["prn"])
            for prn in prns:
                data_prn_1: pd.DataFrame = data[0][data[0]["prn"] == prn]
                mean1 = data_prn_1["diff_stec"].mean()
                std1 = data_prn_1["diff_stec"].std()
                mean2 = (data_prn_1[abs(data_prn_1["diff_stec"] - mean1) < std1])["diff_stec"].mean()
                print(f"bogi__prn{prn:0=2}__mean2={mean2}__mean1={mean1}")
                plt.plot(data_prn_1["time"], data_prn_1["diff_stec"], linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)
                plt.savefig(os.path.join(save_directory, f"prn{prn:0=2}.svg"))
                plt.close()


def _plot_events_for_prns(data: pd.DataFrame):
    prns_set = np.unique(data.loc[:,"prn"])
    values_list = []
    for prn in prns_set:
        temp_data = data.loc[data.loc[:,"prn"] == prn, "avrg_diff_stec"]
        values_list.append(temp_data)
    figure: fig.Figure = plt.figure(layout="tight")
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    axes1.eventplot(positions=values_list, lineoffsets=prns_set, orientation="vertical")
    plt.savefig(r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_prns.svg")
    plt.savefig(
        r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_prns.png", dpi=300)
    plt.close(figure)

def _plot_events_for_lats(data: pd.DataFrame):
    lats_sets = range(-90, 90)
    values_list = []
    for lat in lats_sets:
        mask = np.logical_and(data.loc[:,"gdlatr"] >= lat, data.loc[:,"gdlatr"] < lat + 1)
        temp_data = data.loc[mask, "avrg_diff_stec"]
        values_list.append(temp_data)
    figure: fig.Figure = plt.figure(layout="tight")
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    axes1.eventplot(positions=values_list, lineoffsets=lats_sets, orientation="vertical")
    plt.savefig(r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_lats.svg")
    plt.savefig(
        r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_lats.png", dpi=300)
    plt.close(figure)


def _plot_events_for_lons(data: pd.DataFrame):
    lons_sets = range(-180, 180)
    values_list = []
    for lon in lons_sets:
        mask = np.logical_and(data.loc[:,"gdlonr"] >= lon, data.loc[:,"gdlonr"] < lon + 1)
        temp_data = data.loc[mask, "avrg_diff_stec"]
        values_list.append(temp_data)
    figure: fig.Figure = plt.figure(layout="tight")
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    axes1.eventplot(positions=values_list, lineoffsets=lons_sets, orientation="vertical")
    plt.savefig(r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_lons.svg")
    plt.savefig(
        r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/events_by_lons.png", dpi=300)
    plt.close(figure)



DIFFERENCE_PATH = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Difference_between_LOS_and_CMN_files/2022-06-04/difference.h5"
def main_plot_stat_form_difference(difference_path=DIFFERENCE_PATH, sites_path=TEST_SITE_FILE):
    print(1, dt.datetime.now())
    with h5py.File(difference_path, "r") as file_difference:
        difference_dataframe = pd.DataFrame(file_difference["Data"]["Table Layout"][:])
    sites_array = np.unique(difference_dataframe.loc[:,"site"])
    with h5py.File(sites_path, "r") as file_sites:
        sites_table: pd.DataFrame = pd.DataFrame(file_sites["Data"]["Table Layout"][:])
    sites_coordinates_dataframe: pd.DataFrame = sites_table.loc[sites_table.loc[:,"gps_site"].isin(sites_array), ["gps_site", "gdlatr", "gdlonr"]].reset_index(drop=True)
    print(2, dt.datetime.now())
    data_from_difference_list = []
    for site in sites_array:
        difference_dataframe_for_site = difference_dataframe.loc[difference_dataframe.loc[:, "site"] == site]
        for prn in np.unique(difference_dataframe_for_site.loc[:, "prn"]):
            avrg_diff_stec = difference_dataframe_for_site.loc[difference_dataframe_for_site.loc[:,"prn"] == prn, "diff_stec"].mean()
            data_from_difference_list.append((site, prn, avrg_diff_stec))
    print(3, dt.datetime.now())
    sites_coordinates_dict = {site:(gdlatr, gdlonr) for (site, gdlatr, gdlonr) in
                              sites_coordinates_dataframe.itertuples(index=False, name=None)}
    print(4, dt.datetime.now())
    data_list = []
    for site, prn, avrg_diff_stec in data_from_difference_list:
        data_list.append((site, prn, avrg_diff_stec, sites_coordinates_dict[site][0], sites_coordinates_dict[site][1]))
    data_dataframe = pd.DataFrame(data=data_list, columns=["site", "prn", "avrg_diff_stec", "gdlatr", "gdlonr"])
    print(5, dt.datetime.now())
    _plot_events_for_prns(data_dataframe)
    _plot_events_for_lats(data_dataframe)
    _plot_events_for_lons(data_dataframe)



    # data_dataframe = pd.concat([sites_coordinates_dataframe, pd.DataFrame(columns=["avrg_stec"])],axis=1)
    # print(data_dataframe)
    # for site in data_dataframe.loc[:,"gps_site"]:
    #     tec_data_for_site = difference_dataframe.loc[difference_dataframe.loc[:"diff_stec"] == site]
    #     avrg_stec= tec_data_for_site.mean()
    #     data_dataframe.loc[data_dataframe.loc[:,"gps_site"] == site, "avrg_stec"] = avrg_stec

if __name__ == "__main__":
    main_plot_stat_form_difference()
