import numpy as np
import pandas as pd

import converting_to_number as conv

def read_cmn_file(filename):
    result = list()

    with open(filename, "r") as reader:
        temp_text = ""
        if temp_text == "" or temp_text.split()[0] != "Jdatet":
            temp_text = reader.readline()
        for line in reader:
            str_julian_date, str_time, str_prn, str_azimuth, str_elevation, str_latitude, str_longitude, str_stec,\
            str_vtec, str_s4 = line.split()
            julian_date, time, prn, azimuth, elevation, latitude, longitude, stec, vtec, s4 = \
                float(str_julian_date), float(str_time), int(str_prn), float(str_azimuth), float(str_elevation),\
                float(str_latitude), float(str_longitude), float(str_stec), float(str_vtec), float(str_s4)
            temp_dict = {"Jdatet": julian_date, "time": time, "PRN": prn, "Az": azimuth, "Ele": elevation,
                         "Lat": latitude, "Lon": longitude, "Stec": stec, "Vtec": vtec, "S4": s4}
            result.append(temp_dict)
    return result


def read_cmn_file_short(filename):
    result = list()

    with open(filename, "r") as reader:
        temp_text = ""
        while temp_text.isspace() or temp_text == "" or temp_text.split()[0] != "Jdatet":
            temp_text = reader.readline()
        for line in reader:
            str_julian_date, str_time, str_prn, str_azimuth, str_elevation, str_latitude, str_longitude, str_stec,\
            str_vtec, str_s4 = line.split()
            time, prn, vtec = conv.conver_cmn_time_to_float(str_time), int(str_prn), float(str_vtec)
            temp_dict = {"time": time, "PRN": prn, "Vtec": vtec}
            result.append(temp_dict)
    return result


def read_cmn_file_short_pd(filename):
    arr_time = list()
    arr_prn = list()
    arr_vtec = list()
    arr_stec = list()
    with open(filename, "r") as reader:
        temp_text = ""
        while temp_text.isspace() or temp_text == "" or temp_text.split()[0] != "Jdatet" or temp_text.split()[0] != "MJdatet":
            temp_text = reader.readline()
        for line in reader:
            str_julian_date, str_time, str_prn, str_azimuth, str_elevation, str_latitude, str_longitude, str_stec,\
            str_vtec, str_s4 = line.split()
            time, prn, vtec, stec = conv.conver_cmn_time_to_float(str_time), int(str_prn), float(str_vtec),\
                                    float(str_stec)
            arr_time.append(time)
            arr_prn.append(prn)
            arr_vtec.append(vtec)
            arr_stec.append(stec)
    outcome_data = {"time": arr_time, "prn": arr_prn, "vtec": arr_vtec, "stec": arr_stec}
    outcome_dataframe = pd.DataFrame(outcome_data)
    return outcome_dataframe


def read_cmn_file_pd(filename):
    arr_jd = []
    arr_time = []
    arr_PRN = []
    arr_az = []
    arr_ele = []
    arr_lat = []
    arr_lon = []
    arr_stec = []
    arr_vtec = []
    arr_s4 = []

    with open(filename, "r") as reader:
        temp_text = "\n"
        while temp_text == "\n" or temp_text == "" or not "Jdate" in temp_text.split()[0]:
            try:
                temp_text = reader.readline()
            except Exception as ex:
                temp_text = ""
        for line in reader:
            str_julian_date, str_time, str_prn, str_azimuth, str_elevation, str_latitude, str_longitude, str_stec,\
            str_vtec, str_s4 = line.split()
            julian_date, time, prn, azimuth, elevation, latitude, longitude, stec, vtec, s4 = \
                float(str_julian_date), float(str_time), int(str_prn), float(str_azimuth), float(str_elevation),\
                float(str_latitude), float(str_longitude), float(str_stec), float(str_vtec), float(str_s4)
            arr_jd.append(julian_date)
            arr_time.append(time)
            arr_PRN.append(prn)
            arr_az.append(azimuth)
            arr_ele.append(elevation)
            arr_lat.append(latitude)
            arr_lon.append(longitude)
            arr_stec.append(stec)
            arr_vtec.append(vtec)
            arr_s4.append(s4)
    outcome_dataframe = pd.DataFrame({"julian_date": arr_jd, "time": arr_time, "PRN": arr_PRN, "azimuth": arr_az,
                                      "elevation": arr_ele, "latitude": arr_lat, "longitude": arr_lon, "stec": arr_stec,
                                      "vtec": arr_vtec, "s4": arr_s4})
    return outcome_dataframe


if __name__ == "__main__":
    print(read_cmn_file_short("BASE167-2020-06-15.Cmn"))