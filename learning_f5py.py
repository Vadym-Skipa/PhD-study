import h5py
import numpy as np

test_file = r"C:\Users\lisov\Downloads\gps200204g.002.hdf5"

check_file = r"/home/vadymskipa/Downloads/site_20200314.001.h5"
check_file2 = r"/home/vadymskipa/Downloads/site_20211010.001.h5"

def test1():
    with h5py.File(test_file, "r") as file:
        print(file.keys())

def test2():
    with h5py.File(test_file, "r") as file:
        print(file["Data"])
        print(file["Data"].keys())

def test3():
    with h5py.File(test_file, "r") as file:
        print(file["Data"])
        print(file["Data"].keys())
        print(file["Data"]["Table Layout"])

def test4():
    with h5py.File(test_file, "r") as file:
        test_dst = file["Data"]["Table Layout"]
        print(test_dst[0]["tec"])

def check_site_in_site_file(check_site, check_site_file):
    file = h5py.File(check_site_file, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    res = False
    np_check_site = np.array([check_site], dtype="S4")[0]
    sites_list = list(sites)
    if np_check_site in sites_list:
        res = True
    return res

def print_all_sites(check_site_file):
    file = h5py.File(check_site_file, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    sites_list = list(sites)
    print(sites_list)
    print(sites.dtype)

if __name__ == "__main__":
    path = r"/home/vadymskipa/Documents/PhD_student/data/data1/site_20230226.001.hdf5"
    path2 = r"/tmp/site_20230226.001.h5.hdf5"
    file = h5py.File(path, "r")
    print(file.keys())