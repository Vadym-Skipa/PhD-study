import reader_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os
import datetime
import learning_pandas as lp

example_of_std = r"D:\Dyplom\TEC data\BASE037-2020-02-06.Std"
example_of_hdf5 = r"D:\PhD_student\tec_data_from_madrigal\gps200206g.002.hdf5"

def smoothing(array):
    new_array = np.full(len(array), 0.0)
    new_array[0] = array[0]
    new_array[1] = array[1]
    new_array[-2] = array[-2]
    new_array[-1] = array[-1]
    for i in range(2, len(array) - 2):
        new_array[i] = 0.125 * array[i-2] + 0.125 * array[i+2] + 0.25 * array[i-1] + 0.25 * array[i+1] + 0.25 * array[i]
    return new_array

def show_plot_without_smothing(path_to_std: str, path_to_hdf5: str):

    data_std = reader_std.read_std_file_without_lat_outcome_nparray(path_to_std)
    dst_hdf5 = lp.get_data_with_array(36, 50, path_to_hdf5)

    ut1_unix_to_time = lambda item: (item % 86400) / 3600
    ut_array_for_ut1_unix = np.fromiter((ut1_unix_to_time(item) for item in dst_hdf5["ut1_unix"]), dtype=float)

    data_hdf5 = np.array((ut_array_for_ut1_unix, dst_hdf5["tec"]))
    plt.plot(data_std[0], data_std[1], color="red")
    plt.plot(data_hdf5[0], data_hdf5[1], color="green")
    plt.show()

def show_plot_with_smothing(path_to_std: str, path_to_hdf5: str):

    data_std = reader_std.read_std_file_without_lat_outcome_nparray(path_to_std)
    dst_hdf5 = lp.get_data_with_array(36, 50, path_to_hdf5)

    ut1_unix_to_time = lambda item: (item % 86400) / 3600
    ut_array_hdf5 = np.fromiter((ut1_unix_to_time(item) for item in dst_hdf5["ut1_unix"]), dtype=float)
    tec_array_hdg5 = smoothing(dst_hdf5["tec"])

    data_hdf5 = np.array((ut_array_hdf5, tec_array_hdg5))
    plt.plot(data_std[0], data_std[1], color="red")
    plt.plot(data_hdf5[0], data_hdf5[1], color="green")
    plt.show()

if __name__ == "__main__":
    show_plot_with_smothing(example_of_std, example_of_hdf5)