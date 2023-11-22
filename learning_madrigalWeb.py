import madrigalWeb.madrigalWeb as mad
import datetime

cedar_url = r"http://cedar.openmadrigal.org/"

def test1():
    my_data = mad.MadrigalData(cedar_url)
    my_list = my_data.getExperiments(0, 2020, 2, 6, 0, 0, 0, 2020, 2, 7, 0, 0, 0, 1)
    for data in my_list:
        print(data.realUrl, data.instcode, data.name)

def test2():
    my_data = mad.MadrigalData(cedar_url)
    my_list = my_data.getExperiments(8000, 2020, 2, 6, 0, 1, 0, 2020, 2, 6, 23, 59, 59, 1)
    for data in my_list:
        print(data.realUrl, data.instcode, data.name)

if __name__ == "__main__":
    print(datetime.datetime.now())
    test2()
    print(datetime.datetime.now())