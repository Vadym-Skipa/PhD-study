import datetime
import h5py
import numpy
import numpy as np
import os

test_los_file_path = r"/home/vadymskipa/Downloads/los_20211010.001.h5"
test_cmn_path = r"D:\Dyplom\TEC data\BASE035-2020-02-04.Cmn"
test_save_from_cmn_path = r"D:\PhD_student\tec_data_from_cmn"
test_los_from_cmn_path = r"D:\PhD_student\tec_data_from_cmn\los_from_cmn_20200204.001.h5"
test_path_with_cmn = r"D:\Dyplom\TEC data"

#[('year', '<i8'), ('month', '<i8'), ('day', '<i8'), ('hour', '<i8'), ('min', '<i8'), ('sec', '<i8'), ('recno', '<i8'),
# ('kindat', '<i8'), ('kinst', '<i8'), ('ut1_unix', '<f8'), ('ut2_unix', '<f8'), ('pierce_alt', '<f8'),
# ('gps_site', 'S4'), ('sat_id', '<i8'), ('gnss_type', 'S8'), ('gdlatr', '<f8'), ('gdlonr', '<f8'), ('los_tec', '<f8'),
# ('dlos_tec', '<f8'), ('tec', '<f8'), ('azm', '<f8'), ('elm', '<f8'), ('gdlat', '<f8'), ('glon', '<f8'),
# ('rec_bias', '<f8'), ('drec_bias', '<f8')]

class MeasurementException(Exception):
    pass


def get_site_data(madrigal_file, site_arr, site):
    with h5py.File(madrigal_file, "r") as f:
        site_indices = site_arr == site
        site_data = f["Data"]["Table Layout"][site_indices]
        return site_data

def test1_reading_site_by_site():
    t = datetime.datetime.now()
    f = h5py.File(test_los_file_path, "r")
    site_arr = f["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    f.close()
    site = sites[0]
    print(f"getting data from site {str(site)}")
    site_data = get_site_data(test_los_file_path,site_arr, site)
    print(f"took {datetime.datetime.now() - t} secs to get {len(site_data)} measurements from the first site")

    t = datetime.datetime.now()
    site = sites[1]
    print(f"getting data from site {str(site)}")
    site_data = get_site_data(test_los_file_path, site_arr, site)
    print(f"took {datetime.datetime.now() - t} secs to get {len(site_data)} measurements from the second site")

    print("the following are the column names and data types in this file:")
    for col_name, col_type in site_data.dtype.fields.items():
        print(f"{col_name}\t{col_type}")
    print("for example, here are the first four rows:")
    for i in range(4):
        print(i, site_data[i])
    print("for example, here the first four rows of line of sight TEC:")
    for i in range(4):
        print(i, site_data["los_tec"][i])

    print(f"there are {len(sites)} unique sites in the file")


def get_time_data(madrigal_file, time_arr, unix_time):
    with h5py.File(madrigal_file, "r") as f:
        time_indices = time_arr == unix_time
        site_data = f["Data"]["Table Layout"][time_indices]
        return site_data

def test2_reading_time_by_time(path: str):
    t = datetime.datetime.now()
    f = h5py.File(path, "r")
    time_arr = f["Data"]["Table Layout"]["ut1_unix"]
    times = np.unique(time_arr)
    f.close()
    time = times[0]
    print(f"getting data from time {str(time)}")
    time_data = get_site_data(path, time_arr, time)
    print(f"took {datetime.datetime.now() - t} secs to get {len(time_data)} measurements from the first time")

    t = datetime.datetime.now()
    time = times[1]
    print(f"getting data from time {str(time)}")
    time_data = get_site_data(path, time_arr, time)
    print(f"took {datetime.datetime.now() - t} secs to get {len(time_data)} measurements from the second time")

    print("the following are the column names and data types in this file:")
    for col_name, col_type in time_data.dtype.fields.items():
        print(f"{col_name}\t{col_type}")
    print("for example, here are the first four rows:")
    for i in range(4):
        print(i, time_data[i])
    print("for example, here the first four rows of line of sight TEC:")
    for i in range(4):
        print(i, time_data ["los_tec"][i])

    print(f"there are {len(times)} unique times in the file")


def get_time_gps_data(madrigal_file, time_arr, sat_type_arr, unix_time):
    with h5py.File(madrigal_file, "r") as f:
        indices = np.logical_and(time_arr == unix_time, sat_type_arr == b"GPS     ")
        time_data = f["Data"]["Table Layout"][indices]
        return time_data

def test3_reading_time_by_time_filtering_for_GPS_only():
    t = datetime.datetime.now()
    f = h5py.File(test_los_file_path, "r")
    time_arr = f["Data"]["Table Layout"]["ut1_unix"]
    times = np.unique(time_arr)
    try:
        sat_type_arr = f["Data"]["Table Layout"]["gnss_type"]
    except:
        sat_type_arr = numpy.zeros((len(time_arr),), dtype="S8")
        sat_type_arr[:] = "GPS     "
    f.close()
    time_data = get_time_gps_data(test_los_file_path, time_arr, sat_type_arr, times[0])
    print(f"took {datetime.datetime.now() - t} secs to get {len(time_data)} measurements from the first time")

    t = datetime.datetime.now()
    time_data = get_time_gps_data(test_los_file_path, time_arr, sat_type_arr, times[1])
    print(f"took {datetime.datetime.now() - t} secs to get {len(time_data)} measurements from the second time")

    print(f"there are {len(times)} unique times in the file")

def get_columns_of_los_file():
    f = h5py.File(test_los_file_path, "r")
    data = f["Data"]["Table Layout"]
    return data.dtype

def test4_los_file():
    f = h5py.File(test_los_file_path, "r")
    data = f["Data"]["Table Layout"][0:2]
    f.close()
    return data

def test_create_dst():
    f = h5py.File("new_file", "w")
    grp1 = f.create_group("Data")
    grp2 = f.create_group("Metadata")
    dst = grp1.create_dataset("Table Layout", shape=(1,), dtype=get_columns_of_los_file())
    f.close()

def get_number_of_cmn(path: str):
    with open(path, "r") as file:
        i = len(file.readlines())
    return i

def get_attr_from_line_cmn(line: str):
    attrs_list = line.split()
    julian_date, time, prn, azimuth, elevation, latitude, longitude, stec, vtec, s4\
        = float(attrs_list[0]), float(attrs_list[1]), int(attrs_list[2]), float(attrs_list[3]), float(attrs_list[4]), \
          float(attrs_list[5]), float(attrs_list[6]), float(attrs_list[7]), float(attrs_list[8]), float(attrs_list[9])
    return (julian_date, time, prn, azimuth, elevation, latitude, longitude, stec, vtec, s4)

def create_los_arr_from_args(year=1970, month=1, day=1, hour=0, minute=0, sec=0, recno=0, kindat=3505, kinst=8000,
                             ut1_unix=0.0, ut2_unix=0.0, pierce_alt=350.0, gps_site="BASE", sat_id=1,
                             gnss_type="GPS     ", gdlatr=0.0, gdlonr=0.0, los_tec=0.0, dlos_tec=0.0, tec=0.0, azm=0.0,
                             elm=0.0, gdlat=0.0, glon=0.0, rec_bias=0.0, drec_bias=0.0):
    return (year, month, day, hour, minute, sec, recno, kindat, kinst, ut1_unix, ut2_unix, pierce_alt, gps_site, sat_id,
            gnss_type, gdlatr, gdlonr, los_tec, dlos_tec, tec, azm, elm, gdlat, glon, rec_bias, drec_bias)

def get_los_arr_from_cmn_line(line: str, year: int, month: int, day: int, recno: int, res_lat: int, res_lon: int, ):
    attrs = get_attr_from_line_cmn(line)
    if attrs[1] >= 0:
        hour = int(attrs[1] // 1)
        minute = int(((attrs[1] - hour) * 60))
        sec = int(((attrs[1] - hour) * 60 - minute) * 60)
    else:
        raise MeasurementException
    utc_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=sec, microsecond=0)
    utc_unix = utc_dt.timestamp()
    los_arr = create_los_arr_from_args(year=year, month=month, day=day, hour=hour, minute=minute, sec=sec, recno=recno,
                                       ut1_unix=utc_unix, ut2_unix=utc_unix, pierce_alt=350.0, gps_site="BASE",
                                       sat_id=attrs[2], gnss_type="GPS     ", gdlatr=res_lat, gdlonr=res_lon,
                                       los_tec=attrs[7], tec=attrs[8], azm=attrs[3], elm=attrs[4], gdlat=attrs[5],
                                       glon=attrs[6])
    return los_arr

def get_cord_of_cmn_file(path: str):
    with open(path, "r") as file:
        file.readline()
        file.readline()
        res_latitude, res_longitude, res_altitude = file.readline().split()
        res_latitude, res_longitude, res_altitude = float(res_latitude), float(res_longitude), float(res_altitude)
        return res_latitude, res_longitude

def get_date_from_cmn_file_name(filename: str):
    year = int(filename[8:12])
    month = int(filename[13:15])
    day = int(filename[16:18])
    return year, month, day

def get_str_month_or_day(month_or_day: int):
    if month_or_day < 10:
        return "0" + str(month_or_day)
    else:
        return str(month_or_day)

def create_hdf5_file_from_cmn(path_cmn_file: str, path_for_save: str):
    date = get_date_from_cmn_file_name(os.path.split(path_cmn_file)[1])
    los_file_name = "los_from_cmn_" + str(date[0]) + get_str_month_or_day(date[1]) + get_str_month_or_day(date[2]) + \
                    ".001.h5"
    new_hdf5_file = h5py.File(os.path.join(path_for_save, los_file_name), "w")
    grp1 = new_hdf5_file.create_group("Data")
    grp2 = new_hdf5_file.create_group("Metadata")
    number_of_measurements = get_number_of_cmn(path_cmn_file)
    dst = grp1.create_dataset("Table Layout", shape=(number_of_measurements,), dtype=get_columns_of_los_file(),
                              maxshape=(None,))
    resiver_coordinates = get_cord_of_cmn_file(path_cmn_file)
    with open(path_cmn_file, "r") as file:
        while True:
            words = file.readline().split()
            if words and "Jdatet" in words[0]:
                break
        i = 0
        for line in file.readlines():
            try:
                temp_arr = get_los_arr_from_cmn_line(line, date[0], date[1], date[2], i, *resiver_coordinates)
                dst[i] = temp_arr
                i += 1
                if i % 100000 == 1:
                    print(i, datetime.datetime.now())
            except MeasurementException as m:
                pass
        dst.resize((i,))
    new_hdf5_file.close()

def convert_cmn_files_to_hdf5(path_of_cmn_files: str, path_to_save_hdf5_files: str):
    file_name_list = []
    for file in os.listdir(path_of_cmn_files):
        if file.endswith(".Cmn"):
            file_name_list.append(file)
    for file in file_name_list:
        create_hdf5_file_from_cmn(os.path.join(path_of_cmn_files, file), path_to_save_hdf5_files)
        print(f"file - {file} - well done!  {datetime.datetime.now()}")

def check_containing_of_site_in_los_file(los_file_path, check_sites_arr):
    t = datetime.datetime.now()
    print(t)
    file = h5py.File(test_los_file_path, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    print( datetime.datetime.now() - t)
    res_arr = []
    for site in check_sites_arr:
        res = False
        if site in sites:
            res = True
        res_arr.append(res)

    print(res_arr)
    list_of_sites = list(sites)
    print(list_of_sites)

    return res_arr

def get_sites_from_los_by_long_and_lat(path_los_file, min_long, max_long, min_lat, max_lat):
    t = datetime.datetime.now()
    with h5py.File(path_los_file, "r") as file:
        arr_of_lons = file["Data"]["Table Layout"]["gdlonr"]
        print(0, datetime.datetime.now() - t)
        arr_of_lats = file["Data"]["Table Layout"]["gdlatr"]
        print(1, datetime.datetime.now() - t)
        indices1 = np.logical_and(min_long < arr_of_lons, arr_of_lons < max_long)
        print(2, datetime.datetime.now() - t)
        indices2 = np.logical_and(min_lat < arr_of_lats,  arr_of_lats< max_lat)
        print(3, datetime.datetime.now() - t)
        indices = np.logical_and(indices1, indices2)
        print(4, datetime.datetime.now() - t)
        all_data = file["Data"]["Table Layout"][indices]
        print(5, datetime.datetime.now() - t)
        site_arr = all_data["gps_site"]
        print(6, datetime.datetime.now() - t)
    sites = np.unique(site_arr)
    print(list(sites))


def check_shape():
    with h5py.File(r"/home/vadymskipa/Downloads/los_20220604.001.h5", "r") as file:
        print(file["Data"]["Table Layout"].shape)

def check_bogi_site_in_site_file(path_site=r"/home/vadymskipa/Downloads/site_20220604.001.h5"):
    with h5py.File(path_site, "r") as file:
        sites = file["Data"]["Table Layout"]["gps_site"]
    result = False
    if "bogi".encode("ascii") in sites:
        result = True
    return result

if __name__ == "__main__":
    # check_sites = ["chi2", "mon2", "atri"]
    # chi2_coords = (-75, -74, 49, 51)
    # mon2_coords = (-74, -73, 45, 46)
    # atri_coords = (-72, -71, 46, 47)
    # coords = (21, 31, 44, 54)
    # get_sites_from_los_by_long_and_lat(test_los_file_path, *coords)
    print(get_columns_of_los_file())
