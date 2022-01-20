# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:24:12 2022

@author: James Clark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bitcoin = pd.read_csv('BTC_USD_2014-11-03_2021-12-31-CoinDesk-1.csv')

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

# Creating Plot with Position on y-axis
merge.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = merge['Position 1'].plot(secondary_y=True)
plt.title('Moving Averages w/ Position 1')
plt.show()

# 4

# Trading Positions Based on EMA20 and SMA100 with No Short Selling
merge['Position 2']= np.where(ema20 > sma100, 1, 0)

# 5

# Log and Cumulative Returns for First 2 Strategies
merge['Position 1 Log'] = merge['Log Rets'] * merge['Position 1'].shift(1)
merge['P1 Long-Short CumRet'] = np.exp(np.log1p(merge['Position 1 Log']).cumsum())-1

merge['Position 2 Log'] = merge['Log Rets'] * merge['Position 2'].shift(1)
merge['P2 Long Only CumRet'] = np.exp(np.log1p(merge['Position 2 Log']).cumsum())-1

# 6

# Trading Positions Based on VMA20 and SMA100 with Short Selling Allowed
merge['Position 3'] = np.where(vma20 > sma100, 1, -1)

# Creating Plot with Position on y-axis
merge.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
secondary_y = merge['Position 3'].plot(secondary_y=True)
plt.title('Moving Averages w/ Position 3')
plt.show()

# Trading Positions Based on VMA20 and SMA100 with No Short Selling
merge['Position 4']= np.where(vma20 > sma100, 1, 0)

# Log and Cumulative Returns for Second 2 Strategies

merge['Position 3 Log'] = merge['Log Rets'] * merge['Position 3'].shift(1)
merge['P3 Long-Short CumRet'] = np.exp(np.log1p(merge['Position 3 Log']).cumsum())-1

merge['Position 4 Log'] = merge['Log Rets'] * merge['Position 4'].shift(1)
merge['P4 Long Only CumRet'] = np.exp(np.log1p(merge['Position 4 Log']).cumsum())-1

# Output Dataframe to CSV
merge.to_csv('James Clark Assignment 2 Output.csv')

# Outputs
print('The BTC cumulative return is',cum_ret)
print('The cumulative return of the P1 long-short strategy is'
      ,merge['P1 Long-Short CumRet'][-1])
print('The cumulative return of the P2 long only strategy is'
      ,merge['P2 Long Only CumRet'][-1])
print('The cumulative return of the P3 long-short strategy is'
      ,merge['P3 Long-Short CumRet'][-1])
print('The cumulative return of the P4 long only strategy is'
      ,merge['P4 Long Only CumRet'][-1])