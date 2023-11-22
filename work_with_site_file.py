import h5py
import pandas as pd
import numpy as np
import os
import re


def get_sites_by_coordinates(site_file_path, lons, lats):
    with h5py.File(site_file_path, "r") as file_sites:
        sites_table: pd.DataFrame = pd.DataFrame(file_sites["Data"]["Table Layout"][:])
    mask_lon = np.logical_and(sites_table.loc[:, "gdlonr"] <= lons[1],
                              sites_table.loc[:, "gdlonr"] > lons[0])
    mask_lat = np.logical_and(sites_table.loc[:, "gdlatr"] <= lats[1],
                              sites_table.loc[:, "gdlatr"] > lats[0])
    mask = np.logical_and(mask_lat, mask_lon)
    sites_dataframe = sites_table.loc[mask, ["gps_site", "gdlatr", "gdlonr"]]
    return sites_dataframe


def main1():
    lons = (16, 35)
    lats = (45, 55)
    path = r"/home/vadymskipa/Downloads/site_20230226.001.h5"
    sites = get_sites_by_coordinates(path, lons, lats)
    print(sites)


def get_site_file_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory_path, el)) and
                       re.search("^site_(\d{8,8})\S*(\.h5|\.hdf5)$", el))]
    return result_list


def check_site_in_site_file(check_site, check_site_file):
    file = h5py.File(check_site_file, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    res = False
    sites_list = list(sites)
    if check_site in sites_list:
        res = True
    return res


DIRECTORY_PATH1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/"
def main2(directory_path=DIRECTORY_PATH1):
    list_of_site_file_paths = get_site_file_paths_from_directory(directory_path)
    lons = (16, 35)
    lats = (45, 55)
    sites_1 = get_sites_by_coordinates(list_of_site_file_paths[0], lons, lats)
    print(sites_1)
    mask = []
    sites_2 = []
    if len(list_of_site_file_paths) > 1:
        for site in sites_1.loc[:, "gps_site"]:
            flag = True
            for site_file_path in list_of_site_file_paths[1:]:
                if not check_site_in_site_file(site, site_file_path):
                    flag = False
                    break
            mask.append(flag)
        sites_2 = sites_1.loc[mask]
    else:
        sites_2 = sites_1
    print(sites_2)





if __name__ == "__main__":
    res = get_sites_by_coordinates(r"/home/vadymskipa/Documents/PhD_student/data/data1/site_20230228.001.h5.hdf5", (-180, 180), (22, 55))
    print(res)