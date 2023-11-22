import matplotlib.pyplot as plt
import reader_std as rstd
import reader_cmn as rcmn
import numpy as np
import os


TEST_STD = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/STD_data_from_program/bogi155-2022-06-04.Std"
TEST_CMN = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/CMN_data_from_program/bogi154-2022-06-03.Cmn"
SAVE_TEMP_FILES = r"/home/vadymskipa/Documents/Temporary_files"

def draw_first_prn_stec_from_cmn():
    data_cmn = rcmn.read_cmn_file_short_pd(TEST_CMN)
    prns = np.unique(data_cmn["prn"])
    first_prn_data = data_cmn[data_cmn["prn"] == prns[0]]
    plt.plot(first_prn_data["time"], first_prn_data["stec"], linestyle=" ", marker=".", markeredgewidth=1, markersize=1)
    plt.savefig(os.path.join(SAVE_TEMP_FILES, "just_data_points_markeredgewidth.svg"))
    plt.close()
    print(prns[0])


if __name__ == "__main__":
    draw_first_prn_stec_from_cmn()