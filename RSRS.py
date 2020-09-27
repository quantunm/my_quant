import pandas as pd
import statsmodels.api as sm
import numpy as np
import datetime
from enum import Enum
import matplotlib.pyplot as mplt_pyplt
import copy
import log

class Actions(Enum):
    eSell = -1
    eHold = 0
    eBuy = 1


class CRSRS_std:
    def __init__(self):
        self._arr_beta = []
        self._SBuy = 0.0
        self._SSell = 0.0

    def init(self, df, N, threshold_buy, threshold_sell):
        self._arr_beta = []
        seq_high = df.loc[:, 'high'] #column 'hgih'
        seq_low = df.loc[:, 'low']  #column 'low'

        for i in range(len(seq_high))[N:]:
            X = sm.add_constant(seq_low[i - N + 1 : i + 1])
            model = sm.OLS(seq_high[i - N + 1 : i + 1], X)
            results = model.fit()
            self._arr_beta.append(results.params[1])

        self._SBuy = threshold_buy
        self._SSell = threshold_sell

    def run_strategy(self, df, M):
        seq_high = df.loc[:, 'high'] #column 'hgih'
        seq_low = df.loc[:, 'low']  #column 'low'

        X = sm.add_constant(seq_low)
        model = sm.OLS(seq_high, X)
        results = model.fit()

        self._arr_beta.append(results.params[1])
        mu = np.mean(self._arr_beta[-M:])
        sigma = np.std(self._arr_beta[-M:])

        zscore = (results.params[1] - mu) / sigma

        if zscore > self._SBuy:
            return Actions.eBuy, zscore
        elif zscore < self._SSell:
            return Actions.eSell, zscore
        else:
            return Actions.eHold, zscore

    def run_test(self, df_benchmark, initial_capital, commission_rate, stamp_tax_rate, slide_point, 
        threshold_buy, threshold_sell, start_date, end_date, tst_df_FCR, tst_df_BCR, tst_df_NCR, N, M):
        assert(len(tst_df_BCR) == len(tst_df_FCR) == len(tst_df_NCR))

        lg = log.CLog()
        lg.open(tst_df_FCR['ts_code'].iloc[-1], 
            ['datetime', 'action', 'percent', 'price', 'hands', 'fee', 'total_captial', 'profit', 'zscore'])
        df_for_strategy = tst_df_FCR

        df_init = df_for_strategy.loc[:start_date, :]
        df_init = df_init.drop(df_init.index[-1])
        self.init(df_init, N, threshold_buy, threshold_sell)
        init_len = len(df_init)
        df_len = len(df_for_strategy.loc[:end_date,:])
      
        shares = 0.0
        cash = initial_capital
        open_NCR = tst_df_NCR.loc[:, 'open'] #column 'open'
        seq_profit = []
        seq_date = tst_df_NCR.loc[start_date:end_date,:].index
        
        seq_benchmark = df_benchmark.loc[start_date:end_date, 'open']
        seq_benchmark_profit = seq_benchmark/seq_benchmark[0]
        print(len(seq_benchmark_profit), len(seq_date))
        seq_profit.append(1.0)
        hold_period = [None, None]
        seq_hold_period = []
        seq_zscore = [0.0]
        total_capital = initial_capital
        for i in range(init_len, df_len - 1):
            action, zscore = self.run_strategy(df_for_strategy[i - N + 1:i + 1], M)
            seq_zscore.append(zscore)
            if action == Actions.eBuy:
                buy_price_per_hand = open_NCR[i+1] * (1 + slide_point) * 100
                if cash >= buy_price_per_hand:
                    buy_hands = round( cash / buy_price_per_hand)
                    commission = buy_price_per_hand * buy_hands * commission_rate
                    if commission < 5:
                        commission = 5
                    if cash < buy_price_per_hand * buy_hands + commission:
                        buy_hands -= 1
                    commission = buy_price_per_hand * buy_hands * commission_rate
                    if commission < 5:
                        commission = 5
                    cost = buy_price_per_hand * buy_hands + commission
                    if cash >= cost and buy_hands > 0:
                        shares += buy_hands * 100
                        cash -= cost
                        total_capital -= commission
                        hold_period[0] = tst_df_NCR.index[i]
                        lg.write([hold_period[0].strftime('%Y-%m-%d %H:%M:%S'), 'order', cost/(cost + cash), buy_price_per_hand/100,
                            buy_hands, commission, total_capital, 0, zscore])
            elif action == Actions.eSell:
                if shares > 0:
                    earn = open_NCR[i + 1] * ( 1 - slide_point) * shares
                    commission = earn * commission_rate
                    if commission < 5:
                        commission = 5
                    stamp_tax = earn * stamp_tax_rate
                    cash += earn - commission - stamp_tax
                    shares = 0
                    hold_period[1] = tst_df_NCR.index[i]
                    seq_hold_period.append(copy.copy(hold_period))
                    lg.write([hold_period[1].strftime('%Y-%m-%d %H:%M:%S'), 'close', 0, open_NCR[i + 1] * ( 1 - slide_point),
                            -shares/100, commission + stamp_tax, cash, cash - total_capital, zscore])
                    total_capital = cash
                    hold_period = [None, None]
            seq_profit.append((cash + shares * open_NCR[i+1]) / initial_capital)
        
        if hold_period[0] != None:
            if hold_period[1] == None:
                hold_period[1] = tst_df_NCR.index[-1]
            seq_hold_period.append(copy.copy(hold_period))   

        lg.close()  
        
        print(total_capital)   
        
        fig, ax = mplt_pyplt.subplots(2, 1)
        ax[0].plot(seq_date, seq_profit, label= 'strategy profit rate of ' + tst_df_FCR['ts_code'].iloc[-1])
        ax[0].plot(seq_date, seq_benchmark_profit, label= 'benchmark profit of ' + df_benchmark['ts_code'].iloc[-1])
        ax[0].legend(loc='upper left')

        p0 = ax[1].plot(seq_date, tst_df_FCR.loc[start_date:end_date, 'open'], label= tst_df_FCR['ts_code'].iloc[-1])
        for iter in seq_hold_period:
            ax[1].axvspan(iter[0], iter[1], facecolor='#FFBBBB')
        par = ax[1].twinx()
        p1 = par.plot(seq_date, seq_zscore, label='zscore', color='r')
        par.set_ylabel('zscore')
        ax[1].legend(loc='upper left')
        fig.tight_layout()
        mplt_pyplt.show()


class CRSRS_slope:
    def __init__(self):
        self._arr_beta = []

    def init(self, df, N):
        self._arr_beta = []
        seq_high = df.loc[:, 'high'] #column 'hgih'
        seq_low = df.loc[:, 'low']  #column 'low'

        for i in range(len(seq_high))[N:]:
            X = sm.add_constant(seq_low[i - N + 1 : i + 1])
            model = sm.OLS(seq_high[i - N + 1 : i + 1], X)
            results = model.fit()
            self._arr_beta.append(results.params[1])

        mu = np.mean(self._arr_beta)
        sigma = np.std(self._arr_beta)

        self._SBuy = mu + sigma
        self._SSell = mu - sigma

    def run_strategy(self, df, M):
        seq_high = df.loc[:, 'high'] #column 'hgih'
        seq_low = df.loc[:, 'low']  #column 'low'

        X = sm.add_constant(seq_low)
        model = sm.OLS(seq_high, X)
        results = model.fit()

        self._arr_beta.append(results.params[1])
        mu = np.mean(self._arr_beta[-M:])
        sigma = np.std(self._arr_beta[-M:])

        if results.params[1] > mu + sigma:
            return Actions.eBuy, results.params[1]
        elif results.params[1] < mu - sigma:
            return Actions.eSell, results.params[1]
        else:
            return Actions.eHold, results.params[1]

    def run_test(self, df_benchmark, initial_capital, commission_rate, stamp_tax_rate, slide_point, 
        threshold_buy, threshold_sell, start_date, end_date, tst_df_FCR, tst_df_BCR, tst_df_NCR, N, M):
        assert(len(tst_df_BCR) == len(tst_df_FCR) == len(tst_df_NCR))

        lg = log.CLog()
        lg.open(tst_df_FCR['ts_code'].iloc[-1], 
            ['datetime', 'action', 'percent', 'price', 'hands', 'fee', 'total_captial', 'profit', 'slope'])
        df_for_strategy = tst_df_FCR

        df_init = df_for_strategy.loc[:start_date, :]
        df_init = df_init.drop(df_init.index[-1])
        self.init(df_init, N)
        init_len = len(df_init)
        df_len = len(df_for_strategy.loc[:end_date,:])
      
        shares = 0.0
        cash = initial_capital
        open_NCR = tst_df_NCR.loc[:, 'open'] #column 'open'
        seq_profit = []
        seq_date = tst_df_NCR.loc[start_date:end_date,:].index
        
        seq_benchmark = df_benchmark.loc[start_date:end_date, 'open']
        seq_benchmark_profit = seq_benchmark/seq_benchmark[0]
        #print(seq_benchmark_profit)
        seq_profit.append(1.0)
        hold_period = [None, None]
        seq_hold_period = []
        seq_slope = [0.0]
        total_capital = initial_capital
        for i in range(init_len, df_len - 1):
            action, slope = self.run_strategy(df_for_strategy[i - N + 1:i + 1], M)
            seq_slope.append(slope)
            if action == Actions.eBuy:
                buy_price_per_hand = open_NCR[i+1] * (1 + slide_point) * 100
                if cash >= buy_price_per_hand:
                    buy_hands = round( cash / buy_price_per_hand)
                    commission = buy_price_per_hand * buy_hands * commission_rate
                    if commission < 5:
                        commission = 5
                    if cash < buy_price_per_hand * buy_hands + commission:
                        buy_hands -= 1
                    commission = buy_price_per_hand * buy_hands * commission_rate
                    if commission < 5:
                        commission = 5
                    cost = buy_price_per_hand * buy_hands + commission
                    if cash >= cost and buy_hands > 0:
                        shares += buy_hands * 100
                        cash -= cost
                        total_capital -= commission
                        hold_period[0] = tst_df_NCR.index[i]
                        lg.write([hold_period[0].strftime('%Y-%m-%d %H:%M:%S'), 'order', cost/(cost + cash), buy_price_per_hand/100,
                            buy_hands, commission, total_capital, 0, slope])
            elif action == Actions.eSell:
                if shares > 0:
                    earn = open_NCR[i + 1] * ( 1 - slide_point) * shares
                    commission = earn * commission_rate
                    if commission < 5:
                        commission = 5
                    stamp_tax = earn * stamp_tax_rate
                    cash += earn - commission - stamp_tax
                    shares = 0
                    hold_period[1] = tst_df_NCR.index[i]
                    seq_hold_period.append(copy.copy(hold_period))
                    lg.write([hold_period[1].strftime('%Y-%m-%d %H:%M:%S'), 'close', 0, open_NCR[i + 1] * ( 1 - slide_point),
                            -shares/100, commission + stamp_tax, cash, cash - total_capital, slope])
                    total_capital = cash
                    hold_period = [None, None]
            seq_profit.append((cash + shares * open_NCR[i+1]) / initial_capital)
        
        if hold_period[0] != None:
            if hold_period[1] == None:
                hold_period[1] = tst_df_NCR.index[-1]
            seq_hold_period.append(copy.copy(hold_period))   

        lg.close()  
        
        print(total_capital)   
        
        fig, ax = mplt_pyplt.subplots(2, 1)
        ax[0].plot(seq_date, seq_profit, label= 'strategy profit rate of ' + tst_df_FCR['ts_code'].iloc[-1])
        ax[0].plot(seq_date, seq_benchmark_profit, label= 'benchmark profit of ' + df_benchmark['ts_code'].iloc[-1])
        ax[0].legend(loc='upper left')

        p0 = ax[1].plot(seq_date, tst_df_FCR.loc[start_date:end_date, 'open'], label= tst_df_FCR['ts_code'].iloc[-1])
        for iter in seq_hold_period:
            ax[1].axvspan(iter[0], iter[1], facecolor='#FFBBBB')
        par = ax[1].twinx()
        p1 = par.plot(seq_date, seq_slope, label='slope', color='r')
        par.set_ylabel('slope')
        ax[1].legend(loc='upper left')
        fig.tight_layout()
        mplt_pyplt.show()



        