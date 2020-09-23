import os
import csv

class CLog:
    def __init__(self):
        self.__str_log_path = os.getcwd() + '/log/'
        self.__f_csv = None
        self.__csv_writer = None
    def open(self, str_ts_code, head):
        self.__f_csv = open(self.__str_log_path + str_ts_code + '.csv', 'w+', newline="")
        self.__csv_writer = csv.writer(self.__f_csv)
        self.__csv_writer.writerow(head)

    def write(self, arr_info):
        if self.__csv_writer == None:
            return
        self.__csv_writer.writerow(arr_info)
    def close(self):
        if self.__f_csv:
            self.__f_csv.close()
            self.__csv_writer = None