import multiprocessing
import time

def task1():
    time.sleep(2)
    print(2)

def main1():
    process1 = multiprocessing.Process(target=task1)
    for i in range(1):
        process1.start()
    print(1)


def task2(number: float):
    time.sleep(number)
    print(number)
    return number


def test_multiprocessing_pool():
    pool = multiprocessing.Pool(4)
    mylist = []
    for result in pool.map(task2, [9, 9.5, 2.45, 2.5, 1, 7, 1.5, 3]):
        mylist.append(result)
        print("____", result)
    print(mylist)


if __name__ == "__main__":
    test_multiprocessing_pool()


