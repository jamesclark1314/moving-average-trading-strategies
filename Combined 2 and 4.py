# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:08:06 2022

@author: James Clark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

bitcoin = pd.read_csv('BTC_USD_2014-11-03_2021-12-31-CoinDesk-1 (1).csv')

# Set a Datetime Index
bitcoin['Datetime'] = pd.to_datetime(bitcoin['Date'])
bitcoin = bitcoin.set_index(['Datetime'])
del bitcoin['Date']
del bitcoin['Currency']
del bitcoin['24h Open (USD)']
del bitcoin['24h High (USD)']
del bitcoin['24h Low (USD)']

btc_price = bitcoin['Closing Price (USD)']

# 1

# Daily Returns
daily_rets = btc_price.pct_change()
daily_rets = daily_rets.rename('Daily Returns')

# Daily Log Returns
log_rets = np.log(1 + daily_rets)
log_rets = log_rets.rename('Log Returns')

# Cumulative Return
cum_ret = daily_rets.agg(lambda r: (r + 1).prod()- 1)

# 2

# 100-Day Simple Moving Average
sma100 = btc_price.rolling(window = 100).mean()

# Create 20-day SMA to modify price data
sma20 = btc_price.rolling(window = 20).mean()
mod_price = btc_price.copy()
mod_price.iloc[0:20] = sma20[0:20]

# 20-Day Exponential Moving Average
window = 20
ema20 = mod_price.ewm(span = window, adjust = False).mean()

# Creating the Chande Momentum Oscillator
cmo = []

for each_day in range(len(sma20)):
    cmo.append(abs((sum(log_rets.iloc[0+(each_day+1):window+(each_day+1)] > 0)
                  - sum(log_rets.iloc[0+(each_day+1):window+(each_day+1)]
                        <= 0)))/window)
    
# Convert the cmo list into a series    
cmo = pd.Series(cmo)

vma20 = sma20

for i in range(len(sma20)):
    if i < window:
        vma20[i] = sma20[i]
    else:
        vma20[i] = 2/21*cmo.iloc[0+i-window]*btc_price[i]+(1-2/21*cmo.iloc
                                                       [0+i-window])*vma20[i-1]

# Creating a merged dataframe
merge = bitcoin.merge(daily_rets, how = 'left', 
                                  left_index = True, right_index = True)
merge = merge.merge(log_rets, how = 'left', 
                                  left_index = True, right_index = True)
merge = merge.merge(sma100, how = 'left', 
                                  left_index = True, right_index = True)
merge = merge.merge(ema20, how = 'left', 
                                  left_index = True, right_index = True)
merge = merge.merge(vma20, how = 'left', 
                                  left_index = True, right_index = True)
merge.columns = ['Price', 'Daily Rets', 'Log Rets', 'SMA100', 'EMA20', 'VMA20']


# Plot of Annual Time Series of Daily Prices- SMA100- EMA20, VMA20
merge.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
plt.title('Moving Averages')
plt.show()

# 3

# Trading Positions Based on EMA20 and SMA100 with Short Selling Allowed
merge['Position 1'] = np.where(ema20 > sma100, 1, -1)

# 4

# Trading Positions Based on EMA20 and SMA100 with No Short Selling
merge['Position 2']= np.where(ema20 > sma100, 1, 0)

# 5

# Log and Cumulative Returns for First 2 Strategies
merge['Position 1 Log'] = merge['Log Rets'] * merge['Position 1'].shift(1)
merge['P1 Long-Short CumRet'] = np.exp(merge['Position 1 Log'].loc['2015-02-10':].cumsum())

merge['Position 2 Log'] = merge['Log Rets'] * merge['Position 2'].shift(1)
merge['P2 Long Only CumRet'] = np.exp(merge['Position 2 Log'].loc['2015-02-10':].cumsum())

# 6

# Trading Positions Based on VMA20 and SMA100 with Short Selling Allowed
merge['Position 3'] = np.where(vma20 > sma100, 1, -1)


# Trading Positions Based on VMA20 and SMA100 with No Short Selling
merge['Position 4']= np.where(vma20 > sma100, 1, 0)

# Log and Cumulative Returns for Second 2 Strategies
merge['Position 3 Log'] = merge['Log Rets'] * merge['Position 3'].shift(1)
merge['P3 Long-Short CumRet'] = np.exp(merge['Position 3 Log'].loc['2015-02-10':].cumsum())

merge['Position 4 Log'] = merge['Log Rets'] * merge['Position 4'].shift(1)
merge['P4 Long Only CumRet'] = np.exp(merge['Position 4 Log'].loc['2015-02-10':].cumsum())

# Cumulative Returns for a Passive Buy & Hold Strategy
merge['Buy & Hold CumRet'] = np.exp(merge['Log Rets'].loc['2015-02-10':].cumsum())

# Merging the Log Returns with the Bitcoin Price
dataframe = bitcoin.merge(log_rets, how = 'left', 
                                  left_index = True, right_index = True)

dataframe.columns = ['Price', 'Log Rets']

# Designate Ranges
sma_range = range(100,365,14)
ema_range = range(14,92,7)

results = pd.DataFrame(columns = ['EMA Length', 'SMA Length', 'Cum Rets'])

# Compute the cumulative returns for designated range of EMA and SMA windows
# & sort them in decending order, displaying the top 10 strategies.
for ema1,sma1 in product(ema_range, sma_range):
    sma = btc_price.rolling(window = sma1).mean()
    mod_price = btc_price.copy()
    mod_price.iloc[0:ema1] = sma[0:ema1]
    ema = mod_price.ewm(span = ema1, adjust = False).mean()
    dataframe['Position'] = np.where(ema > sma, 1, -1)
    dataframe['pos_log'] = dataframe['Log Rets'] * dataframe['Position'].shift(1)
    cum_rets = np.exp(dataframe['pos_log'].loc['2016-01-01':].sum())-1
    data = {'EMA Length': ema1, 'SMA Length': sma1, 'Cum Rets': cum_rets}
    results = results.append(data, ignore_index = True)
    
print(results.sort_values(by = ['Cum Rets'], ascending = False).head(10))

# Outputs
print("")
print('The cumulative return of a passive buy-and-hold strategy is'
      ,merge['Buy & Hold CumRet'][-1])
print("")
print('The cumulative return of the P1 long-short strategy is'
      ,merge['P1 Long-Short CumRet'][-1])
print("")
print('The cumulative return of the P2 long only strategy is'
      ,merge['P2 Long Only CumRet'][-1])
print("")
print('The cumulative return of the P3 long-short strategy is'
      ,merge['P3 Long-Short CumRet'][-1])
print("")
print('The cumulative return of the P4 long only strategy is'
      ,merge['P4 Long Only CumRet'][-1])

# Separating the Dataframe Into Years
y2015 = merge.loc['2015']
y2016 = merge.loc['2016']
y2017 = merge.loc['2017']
y2018 = merge.loc['2018']
y2019 = merge.loc['2019']
y2020 = merge.loc['2020']
y2021 = merge.loc['2021']

# PLOTS

# Creating Plots with Position 1 on y-axis
y2015.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2015['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2015')
plt.show()

y2016.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2016['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2016')
plt.show()

y2017.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2017['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2017')
plt.show()

y2018.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2018['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2018')
plt.show()

y2019.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2019['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2019')
plt.show()

y2020.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2020['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2020')
plt.show()

y2021.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2021['Position 1'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 1 - 2021')
plt.show()

# Creating Plot with Position 3 on y-axis
y2015.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2015['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2015')
plt.show()

y2016.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2016['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2016')
plt.show()

y2017.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2017['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2017')
plt.show()

y2018.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2018['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2018')
plt.show()

y2019.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2019['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2019')
plt.show()

y2020.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2020['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2020')
plt.show()

y2021.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = y2021['Position 3'].plot(secondary_y = True)
plt.title('Moving Averages w/ Position 3 - 2021')
plt.show()

# Output Dataframe to CSV
merge.to_csv('Moving Average Output.csv')