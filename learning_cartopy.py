import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.mpl.geoaxes as cgeoaxes
from shapely.geometry import Polygon
from shapely.geometry import Point
import cartopy.feature as cfeature
import os
import work2_1 as w21
import pandas as pd
import matplotlib.axes as axs
import matplotlib.figure as fig
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm
import matplotlib as mpl
import datetime as dt
import numpy as np
from collections import namedtuple
from typing import Dict
from matplotlib.transforms import offset_copy
import re


def test1():
    ax: cgeoaxes.GeoAxes = plt.axes(projection=ccrs.LambertConformal())
    ax.coastlines()
    mypolygon = Polygon(shell=((-5, -10), (-5, 20), (15, 20), (15, -10), (-5, -10)))
    ax.add_geometries([mypolygon], crs=ccrs.PlateCarree(), color="red", alpha=0.3)
    # Save the plot by calling plt.savefig() BEFORE plt.show()
    # plt.savefig('coastlines.pdf')
    # plt.savefig('coastlines.png')
    print(ax.collections)
    plt.show()


def plot_city_lights():
    # Define resource for the NASA night-time illumination data.
    base_uri = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer_name = 'VIIRS_CityLights_2012'

    # Create a Cartopy crs for plain and rotated lat-lon projections.
    plain_crs = ccrs.PlateCarree()
    rotated_crs = ccrs.RotatedPole(pole_longitude=120.0, pole_latitude=45.0)

    fig = plt.figure()

    # Plot WMTS data in a specific region, over a plain lat-lon map.
    ax = fig.add_subplot(1, 2, 1, projection=plain_crs)
    ax.set_extent([-6, 3, 48, 58], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='yellow')
    ax.gridlines(color='lightgrey', linestyle='-')
    # Add WMTS imaging.
    ax.add_wmts(base_uri, layer_name=layer_name)

    # Plot WMTS data on a rotated map, over the same nominal region.
    ax = fig.add_subplot(1, 2, 2, projection=rotated_crs)
    ax.set_extent([-6, 3, 48, 58], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='yellow')
    ax.gridlines(color='lightgrey', linestyle='-')
    # Add WMTS imaging.
    ax.add_wmts(base_uri, layer_name=layer_name)

    plt.show()


def test2():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EquidistantConic(central_longitude=36, central_latitude=50))
    ax.set_extent([16, 56, 62, 40], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.show()


SOURCE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/057-2023-02-26/Window_3600_Seconds/"
SOURCE_DIRECTORY2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/057-2023-02-26/Window_7200_Seconds/"


def read_date_directory_with_dtec_files(source_directory=SOURCE_DIRECTORY1):
    list_files = os.listdir(source_directory)
    dataframe = pd.DataFrame()
    for file in list_files:
        path = os.path.join(source_directory, file)
        dataframe = pd.concat([dataframe, w21.read_dtec_file(path)], ignore_index=True)
    max_lat = dataframe.loc[:, "gdlat"].max()
    min_lat = dataframe.loc[:, "gdlat"].min()
    max_lon = dataframe.loc[:, "gdlon"].max()
    min_lon = dataframe.loc[:, "gdlon"].min()
    print(f"lat: {min_lat} - {max_lat}\nlon: {min_lon} - {max_lon}")


def plot_dtec_graph(dataframe: pd.DataFrame, save_path=None, save_name=None, title=None):
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    line4, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["dtec"], linestyle="-", marker=" ",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_ylim(-1.1, 1.1)
    tick1: float = axes1.get_xticks()[0]
    print(tick1)
    tick_width = dt.timedelta(minutes=5)
    number_of_ticks = dt.timedelta(days=(axes1.get_xticks()[-1] - tick1)) // tick_width + 1
    ticks = [tick1 + (tick_width * i).days + (tick_width * i).seconds / (3600 * 24) for i in range(0, number_of_ticks)]
    axes1.set_xticks(ticks)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.show()
    plt.close(figure)


def some_plot(source_directory=SOURCE_DIRECTORY1):
    list_files = os.listdir(source_directory)
    for file in list_files:
        path = os.path.join(source_directory, file)
        temp_dataframe: pd.DataFrame = w21.read_dtec_file(path)
        temp_dataframe = w21.add_timestamp_column_to_df(temp_dataframe, dt.datetime(year=2023, month=2, day=26,
                                                                                    tzinfo=dt.timezone.utc))
        temp_dataframe = w21.add_datetime_column_to_df(temp_dataframe)
        plot_dtec_graph(dataframe=temp_dataframe)


def filter_dataframe_by_min_elm(dataframe: pd.DataFrame, min_elm=30):
    mask = dataframe.loc[:, "elm"] >= min_elm
    dataframe = dataframe[mask].reset_index(drop=True)
    return dataframe


def get_start_timestamp(timestamp, period):
    start_date_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    start_date_datetime = start_date_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date_timestamp = start_date_datetime.timestamp()
    start_timestamp = (timestamp - start_date_timestamp) // period * period + start_date_timestamp
    return start_timestamp


def get_end_timestamp(timestamp, period):
    end_date_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    end_date_datetime = end_date_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_timestamp = end_date_datetime.timestamp()
    end_timestamp = ((timestamp - end_date_timestamp) // period + 1) * period + end_date_timestamp
    return end_timestamp


def count_data_for_tick(dataframe: pd.DataFrame):
    data_dict = {}
    sat_id_array = np.unique(dataframe.loc[:, "sat_id"])
    for sat_id in sat_id_array:
        sat_id_dataframe: pd.DataFrame = dataframe.loc[dataframe.loc[:, "sat_id"] == sat_id]
        mean_dtec = sat_id_dataframe.loc[:, "dtec"].mean()
        mean_gdlat = sat_id_dataframe.loc[:, "gdlat"].mean()
        mean_gdlon = sat_id_dataframe.loc[:, "gdlon"].mean()
        number = len(sat_id_dataframe.index)
        Sat_tuple = namedtuple("Sat_id_tuple", ["mean_dtec", "mean_gdlon", "mean_gdlat", "number"])
        sat_tuple = Sat_tuple(mean_dtec, mean_gdlon, mean_gdlat, number)
        data_dict[sat_id] = sat_tuple
    return data_dict


def plot_sat_dtec(data_dict: Dict, title):
    fig: plt.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300], dpi=300)
    fig.suptitle(title)
    ax: cgeoaxes.GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([21, 51, 57, 35], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    list_x = [sat_tuple.mean_gdlon for sat_tuple in data_dict.values()]
    list_y = [sat_tuple.mean_gdlat for sat_tuple in data_dict.values()]
    list_color = [sat_tuple.mean_dtec for sat_tuple in data_dict.values()]
    list_size = [sat_tuple.number for sat_tuple in data_dict.values()]
    list_sat_id = [sat_id for sat_id in data_dict.keys()]
    trans_offset = offset_copy(ax.transData, fig=fig, x=0.05, y=0.10, units='inches')
    color_normalize = mplcolors.Normalize(-0.6, 0.6)
    colormap = mpl.colormaps["inferno"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.scatter(x=list_x, y=list_y, s=list_size, c=list_color, transform=ccrs.PlateCarree(), cmap=colormap, norm=color_normalize)
    for index in range(len(list_x)):
        ax.text(list_x[index], list_y[index], list_sat_id[index], transform=trans_offset)
    # plt.show()
    return fig


SAVE_DIRECTORY_SOMEPLOT2_2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot2/Window_7200_Seconds"
SAVE_DIRECTORY_SOMEPLOT2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot2/Window_3600_Seconds"
def some_plot2(source_directory=SOURCE_DIRECTORY1, period=300, save_directory=SAVE_DIRECTORY_SOMEPLOT2):
    list_files = os.listdir(source_directory)
    dataframe = pd.DataFrame()
    for file in list_files:
        sat_id = int(os.path.splitext(file)[0][1:])
        path = os.path.join(source_directory, file)
        temp_dataframe: pd.DataFrame = w21.read_dtec_file(path)
        temp_dataframe = temp_dataframe.assign(sat_id=sat_id)
        dataframe = pd.concat([dataframe, temp_dataframe], ignore_index=True)
    dataframe = w21.add_timestamp_column_to_df(dataframe, dt.datetime(year=2023, month=2, day=26,
                                                                                tzinfo=dt.timezone.utc))
    dataframe = w21.add_datetime_column_to_df(dataframe)
    dataframe = filter_dataframe_by_min_elm(dataframe, 20)
    start_timestamp = get_start_timestamp(dataframe.loc[:, "timestamp"].min(), period)
    end_timestamp = get_end_timestamp(dataframe.loc[:, "timestamp"].max(), period)
    list_tick_timestamp = [timestamp for timestamp in range(int(start_timestamp), int(end_timestamp+1), period)]
    tick_dict = {}
    for tick_timestamp in list_tick_timestamp:
        mask = dataframe.loc[:, "timestamp"] >= tick_timestamp
        mask = np.logical_and(mask, dataframe.loc[:, "timestamp"] < (tick_timestamp + period))
        tick_dataframe = dataframe.loc[mask]
        tick_dict[tick_timestamp] = count_data_for_tick(tick_dataframe)
    for tick_timestamp, data_dict in tick_dict.items():
        tick_datetime = dt.datetime.fromtimestamp(tick_timestamp, tz=dt.timezone.utc)
        title = f"{tick_datetime.timetuple().tm_yday:0=3}-{tick_datetime.year}-{tick_datetime.month:0=2}-" \
                f"{tick_datetime.day:0=2}---{tick_datetime.hour:0=2}:{tick_datetime.minute:0=2}:" \
                f"{tick_datetime.second:0=2}"
        fig = plot_sat_dtec(data_dict, title)
        save_path = os.path.join(save_directory, title + ".png")
        fig.savefig(save_path)
        plt.close(fig)


SAVE_DIRECTORY_SOMEPLOT3 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot3/Window_7200_Seconds"
def some_plot3():
    some_plot2(source_directory=SOURCE_DIRECTORY2, period=900, save_directory=SAVE_DIRECTORY_SOMEPLOT3)


def check_date_directory_name(dir_name):
    re_pattern = "^\d{3}-\d{4}-\d{2}-\d{2}$"
    if re.search(re_pattern, dir_name):
        return True
    return False


def check_window_directory_name(dir_name):
    re_pattern = "^Window_\d+_Seconds$"
    if re.search(re_pattern, dir_name):
        return True
    return False


SOURCE_DIRECTORY4 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT"
SAVE_DIRECTORY_SOMEPLOT4 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot4"
def some_plot4(source_directory=SOURCE_DIRECTORY4, period=300, save_directory=SAVE_DIRECTORY_SOMEPLOT4):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            some_plot2(directory_path_1, period, save_directory_path_1)



def plot_diff_from_directory(source_directory, save_directory, name):
    list_entries = os.listdir(source_directory)
    for entry in list_entries:
        name_2 = name + "_" + os.path.splitext(entry)[0]
        entry_path = os.path.join(source_directory, entry)
        dataframe = w21.read_dtec_file(entry_path)
        temp_date = dt.datetime(year=int(name_2[4:8]), month=int(name_2[9:11]),
                                day=int(name_2[12:14]), tzinfo=dt.timezone.utc)
        dataframe = w21.add_timestamp_column_to_df(dataframe, temp_date)
        dataframe = w21.add_datetime_column_to_df(dataframe)
        w21.plot_difference_graph(save_directory, name_2, dataframe)

SAVE_DIRECTORY_PLOT_DIFF1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/plot_diff1"
def plot_diff1(source_directory=SOURCE_DIRECTORY4, save_directory=SAVE_DIRECTORY_PLOT_DIFF1):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        name_0 = os.path.basename(directory_path_0)
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            name_1 = name_0 + "_" + os.path.basename(directory_path_1)
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            plot_diff_from_directory(directory_path_1, save_directory_path_1, name_1)


DEVIATION = 0.3
def plot_difference_graph2(save_path, name, dataframe, title=None):
    if not title:
        title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    temp_dataframe = dataframe.loc[dataframe.loc[:, "dtec"] > DEVIATION]
    line1, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="blue", markeredgewidth=0.8, markersize=0.9)
    temp_dataframe = dataframe.loc[dataframe.loc[:, "dtec"] < -DEVIATION]
    line2, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="red", markeredgewidth=0.8, markersize=0.9)
    temp_mask = np.logical_and(dataframe.loc[:, "dtec"] <= DEVIATION, dataframe.loc[:, "dtec"] >= -DEVIATION)
    temp_dataframe = dataframe.loc[temp_mask]
    line3, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="gray", markeredgewidth=0.5, markersize=0.6)
    start_datetime = dataframe.iloc[0].loc["datetime"].replace(hour=0, minute=0, second=0, microsecond=0)
    end_datetime = start_datetime + dt.timedelta(days=1)
    axes1.set_xlim(start_datetime, end_datetime)
    axes1.set_ylim(-1.5, 1.5)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


def plot_diff_from_directory2(source_directory, save_directory, name):
    list_entries = os.listdir(source_directory)
    for entry in list_entries:
        name_2 = name + "_" + os.path.splitext(entry)[0]
        entry_path = os.path.join(source_directory, entry)
        dataframe = w21.read_dtec_file(entry_path)
        temp_date = dt.datetime(year=int(name_2[4:8]), month=int(name_2[9:11]),
                                day=int(name_2[12:14]), tzinfo=dt.timezone.utc)
        dataframe = w21.add_timestamp_column_to_df(dataframe, temp_date)
        dataframe = w21.add_datetime_column_to_df(dataframe)
        plot_difference_graph2(save_directory, name_2, dataframe)



SAVE_DIRECTORY_PLOT_DIFF2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/plot_diff2"
def plot_diff2(source_directory=SOURCE_DIRECTORY4, save_directory=SAVE_DIRECTORY_PLOT_DIFF2):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        name_0 = os.path.basename(directory_path_0)
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            name_1 = name_0 + "_" + os.path.basename(directory_path_1)
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            plot_diff_from_directory2(directory_path_1, save_directory_path_1, name_1)


if __name__ == "__main__":
   plot_diff2()
