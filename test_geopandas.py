import h5py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
import datetime as dt
import math
import random
import matplotlib.pyplot as plt


TEST_SITE_FILE = r"/home/vadymskipa/Downloads/site_20220604.001.h5"


def main1(site_file=TEST_SITE_FILE):
    with h5py.File(site_file) as file:
        lats_from_file = pd.Series(file["Data"]["Table Layout"].fields("gdlatr")[:], name="latitude")
        lons_from_file = pd.Series(file["Data"]["Table Layout"].fields("gdlonr")[:], name="longitude")
        coordinates_dataframe = pd.concat([lats_from_file, lons_from_file], axis=1)
    list_of_world = [Point(lon, lat) for lon in range(-180, 180) for lat in range(-90, 90)]

    # # 1
    # start1 = dt.datetime.now()
    # data_receiver_density_dataframe = pd.DataFrame({"geometry": list_of_world, "quantity": np.empty(len(list_of_world),
    #                                                                                                 dtype=int)})
    # for index in range(len(data_receiver_density_dataframe.index)):
    #     point: Point = data_receiver_density_dataframe.loc[index, "geometry"]
    #     mask_lon = np.logical_and(point.x < coordinates_dataframe.loc[:, "longitude"],
    #                               coordinates_dataframe.loc[:, "longitude"] <= point.x + 1)
    #     mask_lat = np.logical_and(point.y < coordinates_dataframe.loc[:, "latitude"],
    #                               coordinates_dataframe.loc[:, "latitude"] <= point.y + 1)
    #     mask = np.logical_and(mask_lon, mask_lat)
    #     quantity = np.count_nonzero(mask)
    #     data_receiver_density_dataframe.loc[index, "quantity"] = quantity
    # mask = data_receiver_density_dataframe.loc[:, "quantity"] >= 1
    # data_receiver_density_dataframe = data_receiver_density_dataframe.loc[mask].reset_index(drop=True)
    # print(1.5, dt.datetime.now() - start1)

    # 2
    data_receiver_density_dict = {}
    for index in lats_from_file.index:
        point = Point(math.floor(lons_from_file[index]), math.floor(lats_from_file[index]))
        if point in data_receiver_density_dict.keys():
            data_receiver_density_dict[point] += 1
        else:
            data_receiver_density_dict[point] = 1
    data_receiver_density_dataframe = pd.DataFrame({"geometry": data_receiver_density_dict.keys(),
                                                     "quantity": data_receiver_density_dict.values()})

    for index in data_receiver_density_dataframe.index:
        point = data_receiver_density_dataframe.loc[index, "geometry"]
        polygon = Polygon([(point.x, point.y), (point.x + 1, point.y), (point.x + 1, point.y + 1), (point.x, point.y + 1)])
        data_receiver_density_dataframe.loc[index, "geometry"] = polygon

    data_receiver_density_geodataframe = gpd.GeoDataFrame(data=data_receiver_density_dataframe, geometry="geometry")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    base = world.plot(color='white', edgecolor='black')
    myplot = data_receiver_density_geodataframe.plot(ax=base, cmap="viridis", legend=True, column="quantity")
    plt.savefig(r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/world.png", dpi=300)
    plt.savefig(r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/world.svg")





if __name__ == "__main__":
    main1()