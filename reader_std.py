import converting_to_number as conv
import pandas as pd

def read_std_file_without_lat(filename):
    result = list()

    with open(filename, "r") as reader:
        for line in reader:
            str_universal_time, str_mean_tec, str_standard_deviation, str_latitude = line.split()
            universal_time, mean_tec, standard_deviation, latitude = \
                conv.convert_to_float(str_universal_time), conv.convert_to_float(str_mean_tec),\
                conv.convert_to_float(str_standard_deviation), conv.convert_to_float(str_latitude)
            temp_dict = {"UT": universal_time, "TEC": mean_tec, "Sigma": standard_deviation}
            result.append(temp_dict)

    return result


def read_std_file_without_lat_outcome_nparray(filename):
    result_ut = list()
    result_tec = list()
    result_dtec = list()

    with open(filename, "r") as reader:
        for line in reader:
            str_universal_time, str_mean_tec, str_standard_deviation, str_latitude = line.split()
            universal_time, mean_tec, standard_deviation, latitude = \
                conv.convert_to_float(str_universal_time), conv.convert_to_float(str_mean_tec),\
                conv.convert_to_float(str_standard_deviation), conv.convert_to_float(str_latitude)
            result_ut.append(universal_time)
            result_tec.append(mean_tec)
            result_dtec.append(standard_deviation)

        result = pd.DataFrame({"UT": result_ut, "tec": result_tec, "dtec": result_dtec})
        return result
