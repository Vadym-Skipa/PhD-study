import os
import statistics as stat
import geopandas as gpd
import matplotlib.axes as axs
import matplotlib.figure as fig
import matplotlib.pyplot as plt
import os.path as path_module
import pandas as pd
import numpy as np
import datetime as dt
import read_los_file as rlf
import multiprocessing as mp
import matplotlib as mpl
from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import List, Dict
import h5py
import math

MAIN_DIRECTORY = r"/home/vadymskipa/Documents/PhD_student/Southern_california"
INPUT_DATA_DIRECTORY = r"/home/vadymskipa/Documents/PhD_student/Southern_california/Input_Data"
PLOT_DIRECTORY = r"/home/vadymskipa/Documents/PhD_student/Southern_california/Plots"

TEST_LOS_FILE1 = r"/home/vadymskipa/Downloads/los_20220604.001.h5"
TEST_SITE_FILE1 = r"/home/vadymskipa/Downloads/site_20220604.001.h5"

LONGITUDE_LIMITS = (-124, -110)
LATITUDE_LIMITS = (30, 40)
SPATIAL_RESOLUTION = 0.15
TIME_RESOLUTION = 600
ELM_MIN = 35
MAX_PERIOD_OF_TIME_IN_ONE_OBSERVATION_SEC = 180
PERIOD_OF_GETTING_AVERAGE = 3600
HALF_PERIOD = PERIOD_OF_GETTING_AVERAGE // 2
PERIOD_BETWEEN_TWO_OBSERVATIONS = 30

NUMBER_OF_PROCESS = 8


# the area of 112◦–124◦W and 31◦–40◦N
def plot_map(data: gpd.GeoDataFrame = None, use_cmp=False):
    figure, axes = plt.subplots(layout="tight")
    world_geodataframe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    axes_world: axs.Axes = world_geodataframe.plot(ax=axes, color='white', edgecolor='black')
    axes_world.set_xlim(*LONGITUDE_LIMITS)
    axes_world.set_ylim(*LATITUDE_LIMITS)
    if type(data) != type(None):
        if use_cmp:
            axes_with_data = data.plot(ax=axes_world, cmap="viridis", legend=True, column="quantity")
        else:
            axes_with_data = data.plot(ax=axes_world, color="blue", markersize=2)
    return figure


def save_map(figure: fig.Figure, path):
    plt.figure(figure)
    plt.savefig(path, dpi=300)


def get_sites_of_southern_california(site_file_path):
    with h5py.File(site_file_path, "r") as file_sites:
        sites_table: pd.DataFrame = pd.DataFrame(file_sites["Data"]["Table Layout"][:])
    mask_lon = np.logical_and(sites_table.loc[:, "gdlonr"] <= LONGITUDE_LIMITS[1],
                              sites_table.loc[:, "gdlonr"] > LONGITUDE_LIMITS[0])
    mask_lat = np.logical_and(sites_table.loc[:, "gdlatr"] <= LATITUDE_LIMITS[1],
                              sites_table.loc[:, "gdlatr"] > LATITUDE_LIMITS[0])
    mask = np.logical_and(mask_lat, mask_lon)
    sites_SC_dataframe = sites_table.loc[mask, ["gps_site", "gdlatr", "gdlonr"]]
    return sites_SC_dataframe


def convert_dataframe_to_geodataframe(dataframe: pd.DataFrame):
    list_of_coordinates = []
    for index in dataframe.index:
        coordinate = Point(dataframe.loc[index, "gdlonr"], dataframe.loc[index, "gdlatr"])
        list_of_coordinates.append(coordinate)
    geodataframe = gpd.GeoDataFrame(data=dataframe.loc[:, "gps_site"], geometry=list_of_coordinates)
    return geodataframe


def _round(value, resolution=SPATIAL_RESOLUTION):
    return math.floor(value / resolution) * resolution


def _convert_point_to_polygon(point: Point, resolution=SPATIAL_RESOLUTION):
    polygon = Polygon([(point.x, point.y), (point.x + resolution, point.y),
                       (point.x + resolution, point.y + resolution), (point.x, point.y + resolution)])
    return polygon


def _convert_point_to_bigger_polygon(point: Point, resolution=SPATIAL_RESOLUTION):
    polygon = Polygon([(point.x - 0.5 * resolution, point.y - 0.5 * resolution),
                       (point.x + 1.5 * resolution, point.y - 0.5 * resolution),
                       (point.x + 1.5 * resolution, point.y + 1.5 * resolution),
                       (point.x - 0.5 * resolution, point.y + 1.5 * resolution)])
    return polygon


def _count_density_of_receivers(dataframe: pd.DataFrame, resolution=SPATIAL_RESOLUTION):
    quantity_dict = {}
    for index in dataframe.index:
        point = Point(_round(dataframe.loc[index, "gdlonr"]), _round(dataframe.loc[index, "gdlatr"]))
        if point in quantity_dict.keys():
            quantity_dict[point] += 1
        else:
            quantity_dict[point] = 1

    quantity_dataframe = pd.DataFrame({"geometry": quantity_dict.keys(), "quantity": quantity_dict.values()})
    for index in quantity_dataframe.index:
        point = quantity_dataframe.loc[index, "geometry"]
        polygon = _convert_point_to_bigger_polygon(point)
        quantity_dataframe.loc[index, "geometry"] = polygon
    quantity_geodataframe = gpd.GeoDataFrame(data=quantity_dataframe, geometry="geometry")
    return quantity_geodataframe


def _save_time_column_from_los_file(save_directory_path=None, los_file_path=TEST_LOS_FILE1):
    with h5py.File(los_file_path, "r") as file:
        time_array = file["Data"]["Table Layout"]["ut1_unix"]
    time_series = pd.Series(time_array)
    full_save_path = path_module.join(save_directory_path, "time_dataframe_" +
                                      path_module.splitext(path_module.basename(los_file_path))[0] + ".scv")
    return time_series.to_csv(full_save_path)


def _save_time_column_from_los_file_hdf(save_directory_path=None, los_file_path=TEST_LOS_FILE1):
    with h5py.File(los_file_path, "r") as file:
        time_array = file["Data"]["Table Layout"]["ut1_unix"]
    time_series = pd.Series(time_array)
    full_save_path = path_module.join(save_directory_path, "time_dataframe_binary_" +
                                      path_module.splitext(path_module.basename(los_file_path))[0] + ".hdf5")
    return time_series.to_hdf(full_save_path, mode="w", key="s")


def _save_hour_and_minute_columns_from_los_file_hdf5(save_directory_path=None, los_file_path=TEST_LOS_FILE1):
    with h5py.File(los_file_path, "r") as file:
        hour_array = file["Data"]["Table Layout"]["hour"]
        minute_array = file["Data"]["Table Layout"]["min"]
    hour_series = pd.Series(hour_array)
    minute_series = pd.Series(minute_array)
    # hour_min_dataframe = pd.DataFrame({"hour": hour_array, "min": minute_array})
    hour_full_save_path = path_module.join(save_directory_path, "hour_dataframe_binary_" +
                                           path_module.splitext(path_module.basename(los_file_path))[0] + ".hdf5")
    hour_series.to_hdf(hour_full_save_path, mode="w", key="s")
    minute_full_save_path = path_module.join(save_directory_path, "minute_dataframe_binary_" +
                                             path_module.splitext(path_module.basename(los_file_path))[0] + ".hdf5")
    minute_series.to_hdf(minute_full_save_path, mode="w", key="s")


def _save_hour_columns_from_los_file_hdf5(save_directory_path=None, los_file_path=TEST_LOS_FILE1):
    with h5py.File(los_file_path, "r") as file:
        hour_array = file["Data"]["Table Layout"]["hour"]
    hour_full_save_path = path_module.join(save_directory_path, "hour_dataset_" +
                                           path_module.splitext(path_module.basename(los_file_path))[0] + ".hdf5")
    temp_type = np.dtype([("hour", "i")])
    with h5py.File(hour_full_save_path, "w") as file:
        dataset = file.create_dataset("hour", shape=(len(hour_array),), dtype=temp_type)
        dataset["hour"] = hour_array


def _save_time_column_from_los_file_hdf5(save_directory_path=None, los_file_path=TEST_LOS_FILE1):
    with h5py.File(los_file_path, "r") as file:
        time_array = file["Data"]["Table Layout"]["ut1_unix"]
    hour_full_save_path = path_module.join(save_directory_path, "time_dataset_" +
                                           path_module.splitext(path_module.basename(los_file_path))[0] + ".hdf5")
    temp_type = np.dtype([("ut1_unix", "f")])
    with h5py.File(hour_full_save_path, "w") as file:
        dataset = file.create_dataset("ut1_unix", shape=(len(time_array),), dtype=temp_type)
        dataset["ut1_unix"] = time_array


def _read_time_column_from_los_file_hdf5(time_dataset_path):
    with h5py.File(time_dataset_path, "r") as file:
        time_array = np.array(file["ut1_unix"]["ut1_unix"])
    return time_array


def _save_los_file(data: pd.DataFrame, save_path, chunks=None, maxshape=None):
    with h5py.File(save_path, "w") as file:
        grp1 = file.create_group("Data")
        grp2 = file.create_group("Metadata")
        data_dtypes = [("hour", "i"), ("min", "i"), ("sec", "i"), ("gps_site", "S4"), ("sat_id", "i"), ("los_tec", "f"),
                       ("tec", "f"),("gdlat", "f"), ("glon", "f")]
        dst = grp1.create_dataset("Table Layout", shape=(len(data),), dtype=data_dtypes, chunks=chunks,
                                  maxshape=maxshape)
        for column in data.columns:
            dst[column] = data.loc[:, column]


def _add_los_file(data: pd.DataFrame, save_path):
    if path_module.exists(save_path):
        with h5py.File(save_path, "r+") as file:
            dst: h5py.Dataset = file["Data"]["Table Layout"]
            dst.resize(dst.shape[0] + data.shape[0], axis=0)
            for column in data.columns:
                dst[-data.shape[0]:, column] = data.loc[:, column]
    else:
        _save_los_file(data, save_path, chunks=True, maxshape=(None,))


def create_los_file_for_sites_in_south_california(save_directory_path=INPUT_DATA_DIRECTORY,
                                                  los_file_path=TEST_LOS_FILE1,
                                                  site_file_path=TEST_SITE_FILE1):
    useful_sites_dataframe: pd.DataFrame = get_sites_of_southern_california(site_file_path).loc[:,
                                           "gps_site"].reset_index(drop=True)
    with h5py.File(los_file_path, "r") as file:
        site_array = file["Data"]["Table Layout"]["gps_site"]
    save_name = "southern_california_useful_" + path_module.basename(los_file_path)
    save_path = path_module.join(save_directory_path, save_name)
    quantity_of_sites = len(useful_sites_dataframe.index)
    for index in useful_sites_dataframe.index:
        temp_site = useful_sites_dataframe.loc[index]
        temp_mask = site_array == temp_site
        print(f"--Reading {index}/{quantity_of_sites} site {dt.datetime.now()}")
        temp_data_dataframe = rlf.get_data_by_indecies_GPS_pd(los_file_path, temp_mask)
        mask_elm = temp_data_dataframe.loc[:, "elm"] > ELM_MIN
        mask_column = ["hour", "min", "sec", "gps_site", "sat_id", "los_tec", "tec", "gdlat", "glon"]
        temp_useful_data: pd.DataFrame = temp_data_dataframe.loc[mask_elm, mask_column]
        temp_useful_data.astype({"gps_site": np.dtype("S")})
        print(f"--Saving {index}/{quantity_of_sites} site {dt.datetime.now()}")
        _add_los_file(temp_useful_data, save_path)


def read_useful_los_file(useful_los_file_path) -> pd.DataFrame:
    with h5py.File(useful_los_file_path, "r") as file:
        useful_data_dateframe = pd.DataFrame(file["Data"]["Table Layout"][:])
    useful_data_dateframe
    sorted_useful_data = useful_data_dateframe.sort_values(by=["gps_site", "sat_id", "hour", "min", "sec"]).reset_index(drop=True)
    return sorted_useful_data


###
###
###
###


def get_time_periods(time_dataframe: pd.Series, max_per=MAX_PERIOD_OF_TIME_IN_ONE_OBSERVATION_SEC,
                     period_of_average=PERIOD_OF_GETTING_AVERAGE) -> List:
    diff_time_dataframe = time_dataframe.diff()
    breaking_index_array = diff_time_dataframe.loc[diff_time_dataframe >= max_per].index
    time_period_list = []
    if not len(breaking_index_array):
        time_period_list.append((time_dataframe.index[0], time_dataframe.index[-1]))
    else:
        start_point = time_dataframe.index[0]
        for index in breaking_index_array:
            time_period_list.append((start_point, index - 1))
            start_point = index
        time_period_list.append((start_point, time_dataframe.index[-1]))
    del_list = []
    for period in time_period_list:
        if (time_dataframe.loc[period[1]] - time_dataframe.loc[period[0]]) < period_of_average:
            del_list.append(period)
    if del_list:
        for period in del_list:
            time_period_list.remove(period)
    return time_period_list


def count_perturbation_values_for_time_period_by_loc(data_for_time_period: pd.DataFrame) -> pd.Series:
    perturbation_values = pd.Series(dtype="f")
    mask = np.logical_and(data_for_time_period["time_sec"] >=
                          (data_for_time_period.iloc[0].loc["time_sec"] + HALF_PERIOD),
                          data_for_time_period["time_sec"] <=
                          (data_for_time_period.iloc[-1].loc["time_sec"] - HALF_PERIOD))
    counting_data_index = data_for_time_period.loc[mask].index
    for index in counting_data_index:
        mean_low = data_for_time_period.loc[
            np.logical_and(data_for_time_period["time_sec"] >=
                           (data_for_time_period.loc[index, "time_sec"] - HALF_PERIOD),
                           data_for_time_period["time_sec"] < data_for_time_period.loc[index, "time_sec"]),
            "tec"
        ].mean()
        mean_high = data_for_time_period.loc[
            np.logical_and(data_for_time_period["time_sec"] <=
                           (data_for_time_period.loc[index, "time_sec"] + HALF_PERIOD),
                        data_for_time_period["time_sec"] > data_for_time_period.loc[index, "time_sec"]),
            "tec"
        ].mean()
        perturbation_values.loc[index] = data_for_time_period.loc[index, "tec"] - (mean_low + mean_high) / 2
    return perturbation_values


def count_perturbation_values_for_time_period_by_index(data_for_time_period: pd.DataFrame) -> pd.Series:
    perturbation_values = pd.Series(dtype="f")
    new_index_data_dataframe = data_for_time_period.copy().assign(id=data_for_time_period.index).set_index("time_sec")
    mask = np.logical_and(data_for_time_period["time_sec"] >= data_for_time_period.iloc[0].loc["time_sec"] + HALF_PERIOD,
                          data_for_time_period["time_sec"] <= data_for_time_period.iloc[-1].loc["time_sec"] - HALF_PERIOD)
    counting_data_index = data_for_time_period.loc[mask].index
    new_indexes = new_index_data_dataframe.index

    def check_up(i):
        for a in range(20):
            if i in new_indexes:
                return i
            i += PERIOD_BETWEEN_TWO_OBSERVATIONS
        raise Exception("CHECK_UP")

    def check_down(i):
        for a in range(20):
            if i in new_indexes:
                return i
            i -= PERIOD_BETWEEN_TWO_OBSERVATIONS
        raise Exception("CHECK_DOWN")

    # counting_data_index = new_index_data_dataframe.loc[check_up(new_indexes[0] + HALF_PERIOD):
    #                       check_down(new_indexes[-1] - HALF_PERIOD), "id"
    #                       ]
    for index in counting_data_index:
        mean_low = new_index_data_dataframe.loc[
                   check_up(data_for_time_period.loc[index, "time_sec"] - HALF_PERIOD):
                   check_down(data_for_time_period.loc[index, "time_sec"] - PERIOD_BETWEEN_TWO_OBSERVATIONS), "tec"
                   ].mean()
        mean_high = new_index_data_dataframe.loc[
                    check_up(data_for_time_period.loc[index, "time_sec"] + PERIOD_BETWEEN_TWO_OBSERVATIONS):
                    check_down(data_for_time_period.loc[index, "time_sec"] + HALF_PERIOD), "tec"
                    ].mean()
        perturbation_values.loc[index] = data_for_time_period.loc[index, "tec"] - (mean_low + mean_high) / 2
    return perturbation_values

def sum_time():
    sum_loc = dt.timedelta(0)
    time = yield
    while time:
        sum_loc += time
        time = yield
    yield sum_loc

sum_loc = sum_time()
sum_index = sum_time()


def count_perturbation_values_for_sat_id(data_for_site_id: pd.DataFrame) -> pd.Series:
    data_for_site_id.assign(time_sec=data_for_site_id["hour"] * 3600 +
                            data_for_site_id["min"] * 60 + data_for_site_id["sec"])
    time_period_list = get_time_periods(data_for_site_id.loc[:, "time_sec"])
    # perturbation_values_series1 = pd.Series(dtype="f")
    # start1 = dt.datetime.now()
    # if time_period_list:
    #     for time_period in time_period_list:
    #         perturbation_values_for_time_period = count_perturbation_values_for_time_period_by_loc(
    #             data_for_site_id.loc[time_period[0]:time_period[1]])
    #         perturbation_values_series1 = pd.concat([perturbation_values_series1, perturbation_values_for_time_period])
    # end1 = dt.datetime.now()

    perturbation_values_series = pd.Series(dtype="f")
    # start2 = dt.datetime.now()
    if time_period_list:
        for time_period in time_period_list:
            perturbation_values_for_time_period = count_perturbation_values_for_time_period_by_index(
                data_for_site_id.loc[time_period[0]:time_period[1]])
            perturbation_values_series = pd.concat([perturbation_values_series, perturbation_values_for_time_period])
    # end2 = dt.datetime.now()

    # print(data_for_site_id.iloc[0].loc["gps_site"], "__", data_for_site_id.iloc[0].loc["sat_id"])
    # loc_time = end1 - start1
    # index_time = end2 - start2
    # sum_loc.send(loc_time)
    # sum_index.send(index_time)
    # print("loc____", loc_time)
    # print("index__", index_time)
    # equal = len(perturbation_values_series.compare(perturbation_values_series1).index)
    # print(equal)
    # if equal:
    #     raise Exception("NOT EQUAL")
    return perturbation_values_series



def count_perturbation_values_for_site(data_for_site: pd.DataFrame) -> pd.Series:
    sat_id_array = np.unique(data_for_site.loc[:, "sat_id"])
    perturbation_values_for_site_series = pd.Series(dtype="f")
    for sat_id in sat_id_array:
        data_for_sat_id = data_for_site.loc[data_for_site.loc[:, "sat_id"] == sat_id]
        perturbation_values_for_sat_id_series = count_perturbation_values_for_sat_id(data_for_sat_id)
        perturbation_values_for_site_series = pd.concat([perturbation_values_for_site_series,
                                                         perturbation_values_for_sat_id_series])
        site = data_for_site.iloc[0].loc["gps_site"]
    return perturbation_values_for_site_series



def add_perturbation_values_to_useful_los_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    perturbation_values_series = pd.Series(dtype="f")
    site_array = np.unique(data.loc[:, "gps_site"])
    i = 1
    max_i = len(site_array)
    start = dt.datetime.now()
    next(sum_loc)
    next(sum_index)
    for site in site_array:
        # if i == 100:
        #     print("RESULT------")
        #     print(f"___ALL___{dt.datetime.now() - start}")
        #     print(f"___LOC___{sum_loc.send(0)}")
        #     print(f"__INDEX__{sum_index.send(0)}")
        #     sum_loc.close()
        #     sum_index.close()
        #     break
        print(f"__________________{i:0=3}/{max_i}____{site}___________")
        i += 1
        data_for_site = data.loc[data.loc[:, "gps_site"] == site]
        perturbation_values_for_site_series = count_perturbation_values_for_site(data_for_site)
        perturbation_values_series = pd.concat([perturbation_values_series, perturbation_values_for_site_series])
    data = data.assign(perturbation=perturbation_values_series)
    return data


def save_los_file_with_perturbation(data: pd.DataFrame, save_path, chunks=None, maxshape=None):
    with h5py.File(save_path, "w") as file:
        grp1 = file.create_group("Data")
        grp2 = file.create_group("Metadata")
        data_dtypes = [("hour", "i"), ("min", "i"), ("sec", "i"), ("time_sec", "i"), ("gps_site", "S4"),
                       ("sat_id", "i"), ("los_tec", "f"), ("tec", "f"), ("perturbation", "f"), ("gdlat", "f"),
                       ("glon", "f")]
        dst = grp1.create_dataset("Table Layout", shape=(len(data),), dtype=data_dtypes, chunks=chunks,
                                  maxshape=maxshape)
        for column in data.columns:
            try:
                dst[column] = data.loc[:, column]
            except ValueError as er:
                print(er)


def main1(save_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1):
    useful_los_file_path = path_module.join(INPUT_DATA_DIRECTORY,
                                            "southern_california_useful_" + path_module.basename(TEST_LOS_FILE1))
    clear_data_dataframe = read_useful_los_file(useful_los_file_path)
    clear_data_with_time_sec = clear_data_dataframe.assign(time_sec=clear_data_dataframe["hour"] * 3600 +
                                                           clear_data_dataframe["min"] * 60 +
                                                           clear_data_dataframe["sec"])
    del clear_data_dataframe
    data_dataframe = add_perturbation_values_to_useful_los_dataframe(clear_data_with_time_sec)
    save_name = "southern_california_useful_data_with_perturbation_values_" + path_module.basename(los_file_path)
    save_path = path_module.join(save_directory_path, save_name)
    save_los_file_with_perturbation(data_dataframe, save_path)


def add_perturbation_values_to_useful_los_dataframe_multiprocessing(data: pd.DataFrame) -> pd.DataFrame:
    perturbation_values_series = pd.Series(dtype="f")
    site_array = np.unique(data.loc[:, "gps_site"])
    max_i = len(site_array)
    pool = mp.Pool(NUMBER_OF_PROCESS)
    for i in range(0, max_i, NUMBER_OF_PROCESS):
        high_limit = i + NUMBER_OF_PROCESS
        if high_limit > max_i:
            high_limit = max_i
        site_dataframe_list = []
        start = dt.datetime.now()
        for site in site_array[i:high_limit]:
            site_dataframe_list.append(data.loc[data.loc[:, "gps_site"] == site].copy())
        for perturbation_values_for_site_series in pool.map(count_perturbation_values_for_site,
                                                            site_dataframe_list):
            perturbation_values_series = pd.concat([perturbation_values_series, perturbation_values_for_site_series])
        print(f"DURABILITY_{i:0=3}-{i+NUMBER_OF_PROCESS-1:0=3}/{max_i}_______{dt.datetime.now() - start}")
    data = data.assign(perturbation=perturbation_values_series)
    return data


def main1_multiprocessing(save_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1):
    print(f"Reading_____{dt.datetime.now()}")
    useful_los_file_path = path_module.join(INPUT_DATA_DIRECTORY,
                                            "southern_california_useful_" + path_module.basename(TEST_LOS_FILE1))
    clear_data_dataframe = read_useful_los_file(useful_los_file_path)
    clear_data_with_time_sec = clear_data_dataframe.assign(time_sec=clear_data_dataframe["hour"] * 3600 +
                                                                    clear_data_dataframe["min"] * 60 +
                                                                    clear_data_dataframe["sec"])
    del clear_data_dataframe
    print(f"Calculating_{dt.datetime.now()}")
    data_dataframe = add_perturbation_values_to_useful_los_dataframe_multiprocessing(clear_data_with_time_sec)
    save_name = "southern_california_useful_data_with_perturbation_values_" + path_module.basename(los_file_path)
    save_path = path_module.join(save_directory_path, save_name)
    print(f"Saving______{dt.datetime.now()}")
    save_los_file_with_perturbation(data_dataframe, save_path)


def add_perturbation_values_to_useful_los_dataframe_multiprocessing_v2(data: pd.DataFrame) -> pd.DataFrame:
    perturbation_values_series = pd.Series(dtype="f")
    site_array = np.unique(data.loc[:, "gps_site"])
    max_i = len(site_array)
    pool = mp.Pool(NUMBER_OF_PROCESS)
    for i in range(0, max_i, NUMBER_OF_PROCESS):
        high_limit = i + NUMBER_OF_PROCESS
        if high_limit > max_i:
            high_limit = max_i
        site_dataframe_list = []
        start = dt.datetime.now()
        for site in site_array[i:high_limit]:
            site_dataframe_list.append(data.loc[data.loc[:, "gps_site"] == site])
        for perturbation_values_for_site_series in pool.map(count_perturbation_values_for_site,
                                                            site_dataframe_list):
            perturbation_values_series = pd.concat([perturbation_values_series, perturbation_values_for_site_series])
        print(f"DURABILITY_{i:0=3}-{i+NUMBER_OF_PROCESS-1:0=3}/{max_i}_______{dt.datetime.now() - start}")
    data = data.assign(perturbation=perturbation_values_series)
    return data


def main1_multiprocessing_v2(save_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1):
    print(f"Reading_____{dt.datetime.now()}")
    useful_los_file_path = path_module.join(INPUT_DATA_DIRECTORY,
                                            "southern_california_useful_" + path_module.basename(TEST_LOS_FILE1))
    clear_data_dataframe = read_useful_los_file(useful_los_file_path)
    clear_data_with_time_sec = clear_data_dataframe.assign(time_sec=clear_data_dataframe["hour"] * 3600 +
                                                                    clear_data_dataframe["min"] * 60 +
                                                                    clear_data_dataframe["sec"])
    del clear_data_dataframe
    print(f"Calculating_{dt.datetime.now()}")
    data_dataframe = add_perturbation_values_to_useful_los_dataframe_multiprocessing_v2(clear_data_with_time_sec)
    save_name = "southern_california_useful_data_with_perturbation_values_" + path_module.basename(los_file_path)
    save_path = path_module.join(save_directory_path, save_name)
    print(f"Saving______{dt.datetime.now()}")
    save_los_file_with_perturbation(data_dataframe, save_path)


def save_plot_perturbation(data, save_path):
    figure: fig.Figure = plt.figure(layout="tight", dpi=300)
    figure.suptitle(f"{path_module.splitext(path_module.basename(save_path))[0]}")
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("VTEC, TEC units")
    line1, = axes1.plot(data["time_sec"], data["tec"], linestyle=" ", marker=".", color="blue", markeredgewidth=1,
                        markersize=1.1)

    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line2, = axes2.plot(data["time_sec"], data["perturbation"], linestyle=" ", marker=".", color="red", markeredgewidth=1,
                        markersize=1.1)

    ytext2 = axes2.set_ylabel("Perturbation_VTEC, TEC units")
    xtext2 = axes2.set_xlabel("Time, sec")
    axes2.set_xlim(*axes1.get_xlim())
    plt.savefig(save_path)
    plt.close(figure)



def save_all_plots_for_site(data: pd.DataFrame, save_directory, str_date=""):
    sat_id_array = np.unique(data.loc[:, "sat_id"])
    for sat_id in sat_id_array:
        save_name = f'{data.iloc[0].loc["gps_site"].decode("ascii")}_{sat_id}_{str_date}.png'
        save_file_path = path_module.join(save_directory, save_name)
        data_for_sat_id = data.loc[data.loc[:, "sat_id"] == sat_id]
        save_plot_perturbation(data_for_sat_id, save_file_path)


def main_plot_perturbation_value(data_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1,
                                 save_directory=PLOT_DIRECTORY):
    str_date = path_module.basename(los_file_path)[4:12]
    print("Reading_______")
    start_read = dt.datetime.now()
    data_file_path = path_module.join(data_directory_path,
                                      "southern_california_useful_data_with_perturbation_values_" +
                                      path_module.basename(los_file_path))
    with h5py.File(data_file_path, "r") as file:
        data = pd.DataFrame(file["Data"]["Table Layout"][:])
    print(f"Read__{dt.datetime.now() - start_read}")
    site_array = np.unique(data.loc[:, "gps_site"])
    save_directory_for_all_plots = path_module.join(save_directory, "Perturbation")
    if not path_module.isdir(save_directory_for_all_plots):
        os.mkdir(save_directory_for_all_plots)
    print("Saving________")
    number_of_sites = len(site_array)
    i = 0
    for site in site_array:
        i += 1
        start_saving = dt.datetime.now()
        save_directory_for_site = path_module.join(save_directory_for_all_plots, site.decode("ascii"))
        if not path_module.isdir(save_directory_for_site):
            os.mkdir(save_directory_for_site)
        data_for_site = data.loc[data.loc[:, "gps_site"] == site]
        save_all_plots_for_site(data_for_site, save_directory_for_site, str_date)
        print(f"DURABILITY_{i:0=3}/{number_of_sites}_______{dt.datetime.now() - start_saving}")


def save_all_plots_for_site2(data_tuple):
    data, save_directory, str_date = data_tuple
    sat_id_array = np.unique(data.loc[:, "sat_id"])
    for sat_id in sat_id_array:
        save_name = f'{data.iloc[0].loc["gps_site"].decode("ascii")}_{sat_id}_{str_date}.png'
        save_file_path = path_module.join(save_directory, save_name)
        data_for_sat_id = data.loc[data.loc[:, "sat_id"] == sat_id]
        save_plot_perturbation(data_for_sat_id, save_file_path)


def main_plot_perturbation_value_multiprocessing(data_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1,
                                                 save_directory=PLOT_DIRECTORY, number_of_proc=NUMBER_OF_PROCESS):
    str_date = path_module.basename(los_file_path)[4:12]
    print("Reading_______")
    start_read = dt.datetime.now()
    data_file_path = path_module.join(data_directory_path,
                                      "southern_california_useful_data_with_perturbation_values_" +
                                      path_module.basename(los_file_path))
    with h5py.File(data_file_path, "r") as file:
        data = pd.DataFrame(file["Data"]["Table Layout"][:])
    print(f"Read__{dt.datetime.now() - start_read}")
    site_array = np.unique(data.loc[:, "gps_site"])
    save_directory_for_all_plots = path_module.join(save_directory, "Perturbation")
    if not path_module.isdir(save_directory_for_all_plots):
        os.mkdir(save_directory_for_all_plots)
    print("Saving________")
    pool = mp.Pool(number_of_proc)
    number_of_sites = len(site_array)
    for index in range(0, number_of_sites, number_of_proc):
        start_saving = dt.datetime.now()
        list_of_data_for_site = []
        for index2 in range(index, index + number_of_proc):
            site = site_array[index]
            save_directory_for_site = path_module.join(save_directory_for_all_plots, site.decode("ascii"))
            if not path_module.isdir(save_directory_for_site):
                os.mkdir(save_directory_for_site)
            data_for_site = data.loc[data.loc[:, "gps_site"] == site]
            list_of_data_for_site.append((data_for_site, save_directory_for_site, str_date))
        pool.map(save_all_plots_for_site2, list_of_data_for_site)
        print(f"DURATION_{index + 1:0=3}-{index + number_of_proc}/{number_of_sites}_______"
              f"{dt.datetime.now() - start_saving}")


def get_maps_dict_from_data_chunk(data: pd.DataFrame) -> Dict:
    number_of_indexes = len(data.index)
    dict_of_maps_dict = {}
    for i in range(number_of_indexes):
        time = _round(data.iloc[i].loc["time_sec"], TIME_RESOLUTION)
        point = Point(_round(data.iloc[i].loc["glon"]), _round(data.iloc[i].loc["gdlat"]))
        if time in dict_of_maps_dict:
            if point in dict_of_maps_dict[time]:
                dict_of_maps_dict[time][point].append(data.iloc[i].loc["perturbation"])
            else:
                dict_of_maps_dict[time][point] = [data.iloc[i].loc["perturbation"]]
        else:
            dict_of_maps_dict[time] = {point: [data.iloc[i].loc["perturbation"]]}
    return dict_of_maps_dict



def get_maps_dict_from_data_multiprocess(data: pd.DataFrame, number_of_processes=NUMBER_OF_PROCESS,
                                         rows_per_turn=10000) -> Dict:
    number_of_indexes = len(data.index)
    pool = mp.Pool(number_of_processes)
    dict_of_maps_dict = {}
    for i in range(0, number_of_indexes, rows_per_turn * number_of_processes):
        start_calculating = dt.datetime.now()
        list_of_data_chunk = []
        high_index = i + number_of_processes * rows_per_turn
        if high_index > number_of_indexes:
            high_index = number_of_indexes
        for x in range(number_of_processes):
            temp_high_index = i + (x + 1) * rows_per_turn
            if temp_high_index > high_index:
                temp_high_index = high_index
                data_chunk = data.iloc[(i + x * rows_per_turn):temp_high_index]
                list_of_data_chunk.append(data_chunk)
                break
            data_chunk = data.iloc[(i + x * rows_per_turn):(i + (x + 1) * rows_per_turn) - 1]
            list_of_data_chunk.append(data_chunk)
        for result in pool.map(get_maps_dict_from_data_chunk, list_of_data_chunk):
            for key1, value1 in result.items():
                if key1 in dict_of_maps_dict:
                    for key2, value2 in value1.items():
                        if key2 in dict_of_maps_dict[key1]:
                            dict_of_maps_dict[key1][key2].extend(value2)
                        else:
                            dict_of_maps_dict[key1][key2] = value2
                else:
                    dict_of_maps_dict[key1] = value1
        print(f"DURATION_{i + 1:0=7}-{high_index:0=7}/{number_of_indexes}_______"
              f"{dt.datetime.now() - start_calculating}")
    return dict_of_maps_dict


def convert_maps_dict_to_geodataframe(maps_dict: Dict) -> gpd.GeoDataFrame:
    dict_for_geodataframe = {}
    for key, value in maps_dict.items():
        polygon = _convert_point_to_polygon(key)
        mean_perturbation = stat.mean(value)
        dict_for_geodataframe[polygon] = mean_perturbation
    result_dataframe = pd.DataFrame({"perturbation": dict_for_geodataframe.values(),
                                     "geometry": dict_for_geodataframe.keys()})
    result_geodataframe = gpd.GeoDataFrame(result_dataframe, geometry="geometry")
    return result_geodataframe


def convert_dict_of_maps_dict_to_dict_of_geodataframe(dict_of_maps_dict: Dict) -> Dict:
    result = {}
    for key, value in dict_of_maps_dict.items():
        result[key] = convert_maps_dict_to_geodataframe(value)
    return result


def save_maps_from_dict_of_geodataframe(dict_of_geodataframe: Dict, save_directory_path):
    for key, value in dict_of_geodataframe.items():
        start = dt.datetime.now()
        figure, axes = plt.subplots(layout="tight", dpi=300)
        hours = key // 3600
        minute = (key - hours * 3600) // 60
        figure.suptitle(f"Perturbation map, {hours:0=2}hours-{minute:0=2}minutes")
        world_geodataframe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        axes_world: axs.Axes = world_geodataframe.plot(ax=axes, color='white', edgecolor='black')
        axes_world.set_xlim(*LONGITUDE_LIMITS)
        axes_world.set_ylim(*LATITUDE_LIMITS)
        axes_with_data = value.plot(ax=axes_world, cmap="plasma", column="perturbation", vmin=-0.8, vmax=0.8, legend=True)
        save_path = path_module.join(save_directory_path, str(key) + ".png")
        plt.savefig(save_path)
        plt.close(figure)
        print(f"Save__{key:0=5}__image__{dt.datetime.now() - start}")


def main_saving_perturbation_maps(data_directory_path=INPUT_DATA_DIRECTORY, los_file_path=TEST_LOS_FILE1,
                                  save_directory=PLOT_DIRECTORY, number_od_proc=NUMBER_OF_PROCESS):
    str_date = path_module.basename(los_file_path)[4:12]
    print("Reading_______")
    start_read = dt.datetime.now()
    data_file_path = path_module.join(data_directory_path,
                                      "southern_california_useful_data_with_perturbation_values_" +
                                      path_module.basename(los_file_path))
    with h5py.File(data_file_path, "r") as file:
        data = pd.DataFrame(file["Data"]["Table Layout"][:])
    print(f"Read__{dt.datetime.now() - start_read}")
    useful_data = data.loc[data["perturbation"].notna()]
    dict_of_maps_dict = get_maps_dict_from_data_multiprocess(useful_data)
    print(f"Calculating___")
    start_calculate = dt.datetime.now()
    dict_of_maps_geodataframe = convert_dict_of_maps_dict_to_dict_of_geodataframe(dict_of_maps_dict)
    print(f"Calculate__{dt.datetime.now() - start_calculate}")
    save_directory1 = path_module.join(save_directory, "Perturbation map")
    if not path_module.isdir(save_directory1):
        os.mkdir(save_directory1)
    save_directory_for_maps = path_module.join(save_directory1, str_date)
    if not path_module.isdir(save_directory_for_maps):
        os.mkdir(save_directory_for_maps)
    save_maps_from_dict_of_geodataframe(dict_of_maps_geodataframe, save_directory_for_maps)




def main():
    main_saving_perturbation_maps()


if __name__ == "__main__":
    main()
