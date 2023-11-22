def convert_to_float(input_str):
    result = None
    try:
        result = float(input_str)
    except ValueError as err:
        None
    return result

def conver_cmn_time_to_float(input_str):
    result = float(input_str)
    if result < 0:
        result = 0
    return result

def convert_prn_to_str(prn: int):
    result = ""
    if prn < 10:
        result = "0" + str(prn)
    else:
        result = str(prn)
    return result