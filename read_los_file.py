import numpy as np 
import h5py
import pandas as pd


def get_data_for_one_site_GPS(path: str, site: str):
    with h5py.File(path, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        np_site = np.array([site], dtype="S4")[0]
        indices_of_site = arr_of_sites == np_site
        data_for_site = file["Data"]["Table Layout"][indices_of_site]
    GPS = np.array(["GPS     "], dtype="S8")[0]
    arr_of_GNSS = data_for_site["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = data_for_site[indices_of_GPS]
    return data_outcome


def get_data_for_one_site_GPS_pd(path: str, site: str):
    with h5py.File(path, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        np_site = np.array([site], dtype="S4")[0]
        indices_of_site = arr_of_sites == np_site
        data_for_site = file["Data"]["Table Layout"][indices_of_site]
    GPS = np.array(["GPS     "], dtype="S8")[0]
    arr_of_GNSS = data_for_site["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = pd.DataFrame(data_for_site[indices_of_GPS])
    return data_outcome


def get_data_by_indecies_GPS_pd(path: str, indices):
    with h5py.File(path, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    GPS = "GPS     ".encode("ascii")
    arr_of_GNSS = data["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = pd.DataFrame(data[indices_of_GPS])
    return data_outcome
