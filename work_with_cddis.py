import requests

TEST_OBS_URL = r"https://cddis.nasa.gov/archive/gnss/data/daily/2022/155/22o/"
TEST_NAV_URL = r"https://cddis.nasa.gov/archive/gnss/data/daily/2022/155/22n/"
TEST_SAVE_DIRECTION = r"/home/vadymskipa/Documents/Check_program_GPS_Gopi/NASA_GPS_data/20220604/"

def get_list_of_url(url=TEST_NAV_URL):
    # Adds '*?list' to the end of URL if not included already
    if not url.endswith("*?list"):
        url = url + "*?list"

    # Makes request of URL, stores response in variable r
    r = requests.get(url)

    lines = r.text.split("\n")
    files = []
    for line in lines:
        words = line.split()
        if len(words) == 2:
            files.append(words[0])
    return files

def choose_22n_from_files(files):
    result_files = []
    for file in files:
        if ".22n.gz" in file:
            result_files.append(file)
    return result_files

def choose_22o_from_files(files):
    result_files = []
    for file in files:
        if ".22o.gz" in file:
            result_files.append(file)
    return result_files


def download_file(filename, url=TEST_NAV_URL, savedir=TEST_SAVE_DIRECTION):

    # Makes request of URL, stores response in variable r
    r = requests.get(url + filename)

    # Opens a local file of same name as remote file for writing to
    with open(savedir + filename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1000):
            fd.write(chunk)

    # Closes local file
    fd.close()

if __name__ == "__main__":
    filenames = choose_22o_from_files(get_list_of_url(TEST_OBS_URL))
    for filename in filenames:
        download_file(filename, TEST_OBS_URL)