import h5py
import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

test_file0 = r"D:\PhD_student\tec_data_from_madrigal\gps200222g.002.hdf5"
test_file1 = r"D:\PhD_student\tec_data_from_madrigal\gps200213g.002.hdf5"
test_file2 = r"D:\PhD_student\tec_data_from_madrigal\gps200214g.002.hdf5"
test_file3 = r"D:\PhD_student\tec_data_from_madrigal\gps200215g.002.hdf5"
test_file4 = r"D:\PhD_student\tec_data_from_madrigal\gps200216g.002.hdf5"
test_file5 = r"D:\PhD_student\tec_data_from_madrigal\gps200217g.002.hdf5"
test_file6 = r"D:\PhD_student\tec_data_from_madrigal\gps200218g.002.hdf5"
test_file7 = r"D:\PhD_student\tec_data_from_madrigal\gps200219g.002.hdf5"
test_file8 = r"D:\PhD_student\tec_data_from_madrigal\gps200220g.002.hdf5"
test_file9 = r"D:\PhD_student\tec_data_from_madrigal\gps200221g.002.hdf5"

test_files = (test_file0, test_file1, test_file2, test_file3, test_file4, test_file5, test_file6, test_file7,
              test_file8, test_file9)

def get_pd_dst_from_hdf5(path: str):
    with h5py.File(path, "r") as file:
        dst = pd.DataFrame(np.array(file["Data"]["Table Layout"]))
        return dst

def get_data_with_loc(lon: int, lat: int, path: str):
    dst = get_pd_dst_from_hdf5(path)
    data = dst.loc[dst["glon"] == lon].loc[dst["gdlat"] == lat]
    return data

def get_data_with_array(lon: int, lat: int, path:str):
    dst = h5py.File(path, "r")
    arr_of_lons = dst["Data"]["Table Layout"]["glon"]
    arr_of_lats = dst["Data"]["Table Layout"]["gdlat"]
    indices = np.logical_and(arr_of_lons == lon, arr_of_lats == lat)
    data = dst["Data"]["Table Layout"][indices]
    dst.close()
    return data

def test1_loc_vs_arr(lon: int, lat: int, path: str):
    print("--------------" + path)
    t_start = datetime.datetime.now()
    d1 = get_data_with_loc(lon, lat, path)
    t_stop = datetime.datetime.now()
    print(f"\ntime for loc\t{t_stop - t_start}\n")
    print(type(d1))
    t_start = datetime.datetime.now()
    d2 = get_data_with_array(lon, lat, path)
    t_stop = datetime.datetime.now()
    print(f"\ntime for arr\t{t_stop - t_start}\n")
    print(type(d2))

def test2():
    with h5py.File(test_file1, "r") as file:
        dst = pd.DataFrame(np.array(file["Data"]["Table Layout"]))
        new_dst = dst.loc[dst["glon"] == 36].loc[dst["gdlat"] == 50]
        new_dst["UT"] = new_dst.apply(lambda row: row["hour"] + row["min"] / 60 + row["sec"] / 3600, axis=1)
        print(new_dst.loc[:, ["UT", "tec"]].values)

def test3():
    with h5py.File(test_file0, "r") as file:
        dst = pd.DataFrame(np.array(file["Data"]["Table Layout"]))
        new_dst = dst.loc[dst["glon"] == 36].loc[dst["gdlat"] == 50]
        new_dst["UT"] = new_dst.apply(lambda row: row["hour"] + row["min"] / 60 + row["sec"] / 3600, axis=1)
        ut = new_dst.loc[:, "UT"].values[::10]
        tec = new_dst.loc[:, "tec"].values[::10]
        dtec = new_dst.loc[:, "dtec"].values[::10]
        plt.errorbar(ut, tec, dtec, fmt="None")
        plt.show()

if __name__ == "__main__":
    for item in test_files:
        test1_loc_vs_arr(36, 50, item)