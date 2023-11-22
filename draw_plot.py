import reader_cmn
import reader_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os
import datetime
import read_los_file as wlos



checking_std_path = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/STD_data_from_program/"
checking_madrigal_data_path = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Data_from_madrigal/"
checking_outcome_path = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Outcome_graphs/"
checking_sites = {"atri": (-71, 47),
                  "mon2": (-74, 46),
                  "chi2": (-74, 50),
                  "bogi": (21, 52),
                  "mikl": (32, 47),
                  "polv": (35, 50)}

poltava_graphs = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Poltava_graphs/"
poltava_std_path = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Poltava_STD_files/"
poltava_madrigal_data_path = r"/home/vadymskipa/Documents/PhD_student/tec_data_from_madrigal/"


path_for_own_data = r"D:/Dyplom/TEC data/"
path_for_save_madrigal_data = r"D:/PhD_student/tec_data_from_madrigal/"
path_for_save_image = r"D:/PhD_student/tec_data/"

example_of_std = r"D:\Dyplom\TEC data\BASE035-2020-02-04.Std"
example_of_std2 = r"D:\Dyplom\TEC data\BASE036-2020-02-05.Std"
example_of_hdf5 = r"C:\Users\lisov\Downloads\gps200204g.002.hdf5"
example_of_hdf52 = r"C:\Users\lisov\Downloads\gps200205g.002.hdf5"
save_file = r"D:\PhD_student\tec_data\20200204.png"
save_file2 = r"D:\PhD_student\tec_data\20200205.png"

example_of_hdf5_3 = r"D:\PhD_student\tec_data_from_madrigal\gps200206g.002.hdf5"
example_of_los_file = r"C:\Users\lisov\Downloads\los_20200204.001.h5"


def convert_prn_to_str(prn: int):
    result = ""
    if prn < 10:
        result = "0" + str(prn)
    else:
        result = str(prn)
    return result

def return_pd_dst(hdf5_file, coordinates):
    with h5py.File(hdf5_file, "r") as file:
        dst = pd.DataFrame(np.array(file["Data"]["Table Layout"]))
        new_dst = dst.loc[dst["glon"] == coordinates[0]].loc[dst["gdlat"] == coordinates[1]]
        new_dst["UT"] = new_dst.apply(lambda row: row["hour"] + row["min"] / 60 + row["sec"] / 3600, axis=1)
        return new_dst

def draw_plot(path_std, path_hdf5, path_for_save, coordinates: (0, 0), max_TECU: 12):
    data = reader_std.read_std_file_without_lat_outcome_nparray(path_std)
    ut_array = data[0]
    tec_array = data[1]
    pd_dst = return_pd_dst(path_hdf5, coordinates)
    ut = pd_dst.loc[:, "UT"].values
    tec = pd_dst.loc[:, "tec"].values
    dtec = pd_dst.loc[:, "dtec"].values
    plt.axis([0, 24, 0, max_TECU])
    plt.errorbar(ut, tec, dtec, fmt="None", errorevery=(0, 10), ecolor="Green")
    plt.plot(ut_array, tec_array)
    plt.plot(ut, tec, color="Red")
    plt.savefig(path_for_save, dpi=300)
    plt.close()

def get_str_month_or_day(month_or_day: int):
    if month_or_day < 10:
        return "0" + str(month_or_day)
    else:
        return str(month_or_day)

def main():
    file_name_list = list()
    for file in os.listdir(path_for_own_data):
        if file.endswith(".Std"):
            file_name_list.append(file)
    for file in file_name_list:
        month = int(file[13:15])
        day = int(file[16:18])
        try:
            madrigal_file = "gps20" + get_str_month_or_day(month) + get_str_month_or_day(day) + "g.002.hdf5"
            draw_plot(path_for_own_data + file, path_for_save_madrigal_data + madrigal_file,
                     path_for_save_image + "2020" + get_str_month_or_day(month) + get_str_month_or_day(day) + ".png")
            print(f"saved image for {month} month {day} day {datetime.datetime.now()}")
        except Exception as ex:
            print(ex)
    print("end")





def main1():

    data = reader_std.read_std_file_without_lat_outcome_nparray(example_of_std)
    ut_array = data[0]
    tec_array = data[1]
    pd_dst = return_pd_dst(example_of_hdf5)
    ut = pd_dst.loc[:, "UT"].values
    tec = pd_dst.loc[:, "tec"].values
    dtec = pd_dst.loc[:, "dtec"].values
    plt.axis([0, 24, 0, 12])
    plt.errorbar(ut, tec, dtec, fmt="None", errorevery=(0, 10), ecolor="Green")
    plt.plot(ut_array, tec_array)
    plt.plot(ut, tec, color="Red")
    plt.savefig(save_file)
    plt.close()

def main2():

    data = reader_std.read_std_file_without_lat_outcome_nparray(example_of_std2)
    ut_array = data[0]
    tec_array = data[1]
    pd_dst = return_pd_dst(example_of_hdf52)
    ut = pd_dst.loc[:, "UT"].values
    tec = pd_dst.loc[:, "tec"].values
    dtec = pd_dst.loc[:, "dtec"].values
    plt.errorbar(ut, tec, dtec, fmt="None", errorevery=(0, 10), ecolor="Green")
    plt.plot(ut_array, tec_array)
    plt.plot(ut, tec, color="Red")
    plt.savefig(save_file2)

def draw_plot_hdf5(path_of_hdf5: str, lon: int, lat: int):
    with h5py.File(path_of_hdf5, "r") as file:
        arr_of_lons = file["Data"]["Table Layout"]["glon"]
        arr_of_lats = file["Data"]["Table Layout"]["gdlat"]
        indices = np.logical_and(arr_of_lons == lon, arr_of_lats == lat)
        all_data = file["Data"]["Table Layout"][indices]
        time_data = all_data["ut1_unix"]
        tec_data = all_data["tec"]
    plt.plot(time_data, tec_data)
    plt.show()

def draw_plot_los_hdf5_for_one_site_and_one_sat(path_of_los_hdf5: str):
    start = datetime.datetime.now()
    with h5py.File(path_of_los_hdf5, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        arr_of_sats = file["Data"]["Table Layout"]["sat_id"]
        print(1)
        sites = np.unique(arr_of_sites)
        site = sites[82]
        indices_of_site = arr_of_sites == site
        data_for_site = file["Data"]["Table Layout"][indices_of_site]
        arr_of_sats_for_site = data_for_site["sat_id"]
        sats = np.unique(arr_of_sats_for_site)
        sat = sats[0]
        print(2)
        indices = np.logical_and(arr_of_sites == site, arr_of_sats == sat)
        all_data = file["Data"]["Table Layout"][indices]
        print(3)

    time_data = all_data["ut1_unix"]
    tec_data = all_data["los_tec"]
    plt.plot(time_data, tec_data)
    print(datetime.datetime.now() - start)
    plt.show()

def draw_plot_los_hdf5_for_one_site_and_one_sat2(path_of_los_hdf5: str):
    start = datetime.datetime.now()
    with h5py.File(path_of_los_hdf5, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        sites = np.unique(arr_of_sites)
        site = sites[2]
        indices_of_site = arr_of_sites == site
        data_for_site = file["Data"]["Table Layout"][indices_of_site]
    arr_of_sats_for_site = data_for_site["sat_id"]
    arr_of_sites_for_site = data_for_site["gps_site"]
    arr_of_gnss_types_for_site = data_for_site["gnss_type"]
    print(len(np.unique(arr_of_sites_for_site)))
    sats = np.unique(arr_of_sats_for_site)
    sat = sats[3]
    gnss_types = np.unique(arr_of_gnss_types_for_site)
    print(gnss_types)
    a = (("GPS" in str(item)) for item in gnss_types)
    for item in a:
        print(item)
    gnss_type = None
    for item in gnss_types:
        if "GPS" in str(item):
            gnss_type = item
            break
    if gnss_type:
        indices = np.logical_and(arr_of_sats_for_site == sat, arr_of_gnss_types_for_site == gnss_type)
    else:
        indices = arr_of_sats_for_site == sat
    all_data = data_for_site[indices]
    arr_of_sats_for_sat = all_data["sat_id"]
    print(len(np.unique(arr_of_sats_for_sat)))

    time_data = all_data["ut1_unix"]
    tec_data = all_data["los_tec"]
    arr_of_gnss_types = all_data["gnss_type"]
    gnss_types = np.unique(arr_of_gnss_types)
    plt.plot(time_data, tec_data, ".")
    print(datetime.datetime.now() - start, "---", gnss_types)
    plt.show()

def main_for_checking():
    file_name_list = list()
    for file in os.listdir(checking_std_path):
        if file.endswith(".Std"):
            file_name_list.append(file)
    for file in file_name_list:
        year = int(file[10:12])
        month = int(file[13:15])
        day = int(file[16:18])
        site = file[0:4]
        try:
            coord = checking_sites[site]
            madrigal_file = "gps" + str(year) + get_str_month_or_day(month) + get_str_month_or_day(day) + "g.002.hdf5"
            draw_plot(checking_std_path + file, checking_madrigal_data_path + madrigal_file,
                      checking_outcome_path + site + "20" + str(year) + get_str_month_or_day(month)
                      + get_str_month_or_day(day) + ".png", coord, 25)
            print(f"saved image site {site} for {month} month {day} day {datetime.datetime.now()}")
        except Exception as ex:
            print(ex)
    print("end")

def draw_graphs_for_poltava():
    file_name_list = list()
    for file in os.listdir(poltava_std_path):
        if file.endswith(".Std"):
            file_name_list.append(file)
    for file in file_name_list:
        year = int(file[10:12])
        month = int(file[13:15])
        day = int(file[16:18])
        site = file[0:4]
        if site == "polv":
            try:
                coord = checking_sites[site]
                madrigal_file = "gps" + str(year) + get_str_month_or_day(month) + get_str_month_or_day(day) + "g.002.hdf5"
                draw_plot(poltava_std_path + file, poltava_madrigal_data_path + madrigal_file,
                          poltava_graphs + site + "20" + str(year) + get_str_month_or_day(month)
                          + get_str_month_or_day(day) + ".png", coord)
                print(f"saved image site {site} for {month} month {day} day {datetime.datetime.now()}")
            except Exception as ex:
                print(ex)
    print("end")

def draw_plot_for_one_site_and_one_sat():
    pass

example_of_los_file2 = r"/home/vadymskipa/Downloads/los_20220604.001.h5"
example_of_cmn_file2 = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/CMN_data_from_program/bogi155-2022-06-04.Cmn"
save_path2 = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/Some_graphs/"

def draw_plots_for_one_site_from_los_and_cmn(path_los: str, path_cmn: str, site: str, path_save: str, save_name: str):
    site_data_los = wlos.get_data_for_one_site_GPS(path_los, site)
    site_data_cmn = reader_cmn.read_cmn_file_short_pd(path_cmn)
    prns_los = np.unique(site_data_los["sat_id"])
    prns_cmn = np.unique(site_data_cmn["prn"])

    print(list(prns_los))
    print(list(prns_cmn))

    for prn in prns_los:
        if prn in prns_cmn:
            indices_of_prn_los = site_data_los["sat_id"] == prn
            prn_data_los = site_data_los[indices_of_prn_los]
            indicies_of_prn_cmn = site_data_cmn["prn"] == prn
            prn_data_cmn = site_data_cmn[indicies_of_prn_cmn]
            temp = datetime.datetime(year=2022, month=6, day=4, tzinfo=datetime.timezone.utc).timestamp()
            temp_time_los = [(i - temp) / 3600 for i in prn_data_los["ut1_unix"]]
            plt.plot(temp_time_los, prn_data_los["los_tec"], ".", color="Red")
            plt.plot(prn_data_cmn["time"], prn_data_cmn["stec"], ".", color="Blue")
            plt.savefig(path_save + save_name + "_" + convert_prn_to_str(prn) + "_stec.svg")
            plt.close()

            plt.plot(temp_time_los, prn_data_los["tec"], ".", color="Red")
            plt.plot(prn_data_cmn["time"], prn_data_cmn["vtec"], ".", color="Blue")
            plt.savefig(path_save + save_name + "_" + convert_prn_to_str(prn) + "_vtec.svg")
            plt.close()
            print("---", prn, "---")



if __name__ == "__main__":
    draw_plots_for_one_site_from_los_and_cmn(example_of_los_file2, example_of_cmn_file2, "bogi", save_path2, "bogi2022_06_04")