import data
import pandas
import RSRS
import datetime

if __name__ == '__main__':
    data_source = data.CDataSource()
    #data_source.download_stock_daily('000001.SZ')
    df_arr = data_source.load_stock_daily('000001.SZ')
    df_benchmark = data_source.load_index_daily('000300.SH')
    df_NCR = df_arr[0]
    df_FCR = df_arr[1]
    df_BCR = df_arr[2]

    rsrs = RSRS.CRSRS_std()
    #rsrs = RSRS.CRSRS_slope()
    rsrs.run_test(df_benchmark, 20000, 0.0003, 0.001, 0.0, 0.7, -0.7, 
        '20100104', '20200914',df_FCR, df_BCR, df_NCR, 18, 600)
    
    #rsrs.run_test(df_benchmark, 20000, 0.0, 0.0, 0.0, 0.7, -0.7, 
    #    '20150105', '20200914',df_FCR, df_BCR, df_NCR, 18, 600)

    #rsrs.run_test(df_benchmark, 2000000, 0.0003, 0.001, 0.0, 0.7, -0.7, 
    #    '20100104', '20200914',df_benchmark, df_benchmark, df_benchmark, 18, 600)

    #rsrs.run_test(df_benchmark, 2000000, 0.0, 0.0, 0.0, 0.7, -0.7, 
    #    '20150105', '20200914',df_benchmark, df_benchmark, df_benchmark, 18, 600)

