import tushare as ts
import os
import pandas as pd
import time

class CDataSource:

    def __init__(self):
        self.__str_data_path = os.getcwd() + '/data/'
        ts.set_token('39474178edd3f4144b29efadf67ff8d2dba42a4bf09c6fb64041c0de')

    def download_stock_daily(self, str_ts_code):
        df_NCR = ts.pro_bar(ts_code = str_ts_code)
        df_FCR = ts.pro_bar(ts_code = str_ts_code, adj='qfq')
        df_BCR = ts.pro_bar(ts_code = str_ts_code, adj='hfq')

        df_NCR = df_NCR.iloc[::-1]
        df_FCR = df_FCR.iloc[::-1]
        df_BCR = df_BCR.iloc[::-1]

        str_name = df_NCR['ts_code'][0]
        df_NCR.to_csv(self.__str_data_path + str_name +'_NCR.csv', index = False)
        df_FCR.to_csv(self.__str_data_path + str_name +'_FCR.csv', index = False)
        df_BCR.to_csv(self.__str_data_path + str_name +'_BCR.csv', index = False)

    def load_stock_daily(self, str_ts_code):
        df_NCR = pd.read_csv(self.__str_data_path + str_ts_code + '_NCR.csv', parse_dates=['trade_date'], index_col=['trade_date'])
        df_FCR = pd.read_csv(self.__str_data_path + str_ts_code + '_FCR.csv', parse_dates=['trade_date'], index_col=['trade_date'])
        df_BCR = pd.read_csv(self.__str_data_path + str_ts_code + '_BCR.csv', parse_dates=['trade_date'], index_col=['trade_date'])

        df_NCR = df_NCR.dropna()
        df_FCR = df_FCR.dropna()
        df_BCR = df_BCR.dropna()
        return self.fit_dataframe_length([df_NCR, df_FCR, df_BCR])
        

    def download_index_daily(self, str_ts_code):
        pro = ts.pro_api()
        df = pro.index_daily(ts_code = str_ts_code)
        df = df.iloc[::-1]
        df.to_csv(self.__str_data_path + str_ts_code + '.csv', index = False)

    def load_index_daily(self, str_ts_code):
        return pd.read_csv(self.__str_data_path + str_ts_code + '.csv', parse_dates=['trade_date'], index_col=['trade_date'])

    def download_stock_1min(self, str_ts_code):
        df_FCR = ts.pro_bar(ts_code=str_ts_code, asset='E',adj='qfq',freq='1min')
        time.sleep(62)
        df_BCR = ts.pro_bar(ts_code=str_ts_code, asset='E',adj='hfq',freq='1min')
        time.sleep(62)
        df_NCR = ts.pro_bar(ts_code=str_ts_code, asset='E',freq='1min')

        df_NCR = df_NCR.iloc[::-1]
        df_FCR = df_FCR.iloc[::-1]
        df_BCR = df_BCR.iloc[::-1]

        str_name = df_NCR['ts_code'][0]
        df_FCR.to_csv(self.__str_data_path + str_name +'_1min_NCR.csv', index = False)
        df_FCR.to_csv(self.__str_data_path + str_name +'_1min_FCR.csv', index = False)
        df_BCR.to_csv(self.__str_data_path + str_name +'_1min_BCR.csv', index = False)

    def fit_dataframe_length(self, arr_df):
        start_date = None
        end_date = None
        for df in arr_df:
            start_date_tmp = df.index[0]
            end_date_tmp = df.index[-1]
            if start_date == None:
                start_date = start_date_tmp
            else:
                start_date = start_date if start_date.__ge__(start_date_tmp) else start_date_tmp

            if end_date == None:
                end_date = end_date_tmp
            else:
                end_date = end_date if end_date.__le__(end_date_tmp) else end_date_tmp
        str_start_date = start_date.strftime('%Y%m%d')
        str_end_date = end_date.strftime('%Y%m%d')   
        output_arr = []
        for df in arr_df:
            output_arr.append(df.loc[str_start_date:str_end_date, :])
        return output_arr 


if __name__ =='__main__':
    '''
    data_source = CDataSource()
    data_source.download_stock_daily('000001.SZ')
    df_arr = data_source.load_stock_daily('000001.SZ')
    for df in df_arr:
        print(df)
    '''
    pass
