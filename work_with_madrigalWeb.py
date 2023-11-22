import madrigalWeb.madrigalWeb as mad
import datetime

cedar_url = r"http://cedar.openmadrigal.org/"

def get_world_wide_tec_from_gps_glonass_for_year(year: int):
    code_for_world_wide_tec_from_gps_glonass = code = 8000
    data_from_madrigal = mad.MadrigalData(cedar_url)
    list_of_experiments = data_from_madrigal.getExperiments(8000, year, 1, 1, 0, 0, 1, year, 12, 31, 23, 59, 59, 1)
    return list_of_experiments

def get_day_experiment_from_list_of_year(list_of_year: list, month: int, day: int):
    for experiment in list_of_year:
        if experiment.startmonth == month:
            if experiment.startday == day:
                return experiment

def download_experiment(month: int, day: int, list_of_experiments: list, path_for_download: str):
    experiment = get_day_experiment_from_list_of_year(list_of_experiments, month, day)
    maddata = mad.MadrigalData(cedar_url)
    maddata.downloadFile(maddata.getExperimentFiles(experiment.id)[0].name, path_for_download, "Vadym Skipa",
                         "lisove2012@gmail.com", "None", format="hdf5")

def test_download():
    list_of_experiments = get_world_wide_tec_from_gps_glonass_for_year(2020)
    experiment = get_day_experiment_from_list_of_year(list_of_experiments, 2, 4)
    maddata = mad.MadrigalData(cedar_url)
    path_for_download = r"D:/PhD_student/tec_data_from_madrigal/"
    maddata.downloadFile(maddata.getExperimentFiles(experiment.id)[0].name, path_for_download, user_fullname="Vadym Skipa",
                         user_email="lisove2012@gmail.com", user_affiliation="None", format="hdf5")


if __name__ == "__main__":
    test_download()