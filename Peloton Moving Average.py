# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:51:30 2022

@author: James Clark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import yfinance as yf

# BACKTEST

# Import Peloton's historical data
peloton = yf.Ticker('PTON')
pton_hist = peloton.history(period = 'max')

# Drop unnecessary columns
del pton_hist['Dividends']
del pton_hist['Stock Splits']
del pton_hist['Volume']
del pton_hist['Low']
del pton_hist['High']
del pton_hist['Open']

pton_hist.columns = ['Close Price']

# Calculate daily returns
pton_hist['Daily Rets'] = pton_hist['Close Price'].pct_change()

# Calculate log returns
pton_hist['Log Rets'] = np.log(1 + pton_hist['Daily Rets'])

# Calculate cumulative returns
cum_ret = pton_hist['Daily Rets'].agg(lambda r: (r + 1).prod()- 1)

# 100-Day Simple Moving Average
sma100 = pton_hist['Close Price'].rolling(window = 100).mean()

# Create 20-day SMA to modify price data
sma20 = pton_hist['Close Price'].rolling(window = 20).mean()
mod_price = pton_hist['Close Price'].copy()
mod_price.iloc[0:20] = sma20[0:20]

# 20-Day Exponential Moving Average
window = 20
ema20 = mod_price.ewm(span = window, adjust = False).mean()

# Creating the Chande Momentum Oscillator
cmo = []

for i in range(len(sma20)):
    cmo.append(abs((sum(pton_hist['Log Rets'].iloc[0+(i+1):window+(i+1)] > 0)
                  - sum(pton_hist['Log Rets'].iloc[0+(i+1):window+(i+1)]
                        <= 0)))/window)
    
# Convert the cmo list into a series    
cmo = pd.Series(cmo)

vma20 = sma20

for i in range(len(sma20)):
    if i < window:
        vma20[i] = sma20[i]
    else:
        vma20[i] = 2/21*cmo.iloc[0+i-window]*pton_hist['Close Price'][i]+(1-2/21*cmo.iloc
                                                       [0+i-window])*vma20[i-1]
        
# Creating a dataframe for the moving averages
mov_av = pd.DataFrame(pton_hist['Close Price'])
mov_av = mov_av.merge(sma100, how = 'left', 
                                  left_index = True, right_index = True)
mov_av = mov_av.merge(ema20, how = 'left', 
                                  left_index = True, right_index = True)
mov_av = mov_av.merge(vma20, how = 'left', 
                                  left_index = True, right_index = True)
mov_av.columns = ('Price','SMA100', 'EMA20', 'VMA20')

# Plot of Annual Time Series of Daily Prices- SMA100- EMA20, VMA20
mov_av.plot(y = ['Price','SMA100', 'EMA20', 'VMA20'])
plt.title('Moving Averages')
plt.show()


# Trading positions based on EMA20 and SMA100
mov_av['Position 1'] = np.where(ema20 > sma100, 1, -1)

# Log and cumulative returns for the first strategy
mov_av['Position 1 Log'] = pton_hist['Log Rets'] * mov_av['Position 1'].shift(1)
mov_av['P1 CumRet'] = np.exp(mov_av['Position 1 Log'].loc['2020-02-19':].cumsum())-1

# Trading positions based on VMA20 and SMA100
mov_av['Position 2'] = np.where(vma20 > sma100, 1, -1)

# Log and cumulative returns for the second strategy
mov_av['Position 2 Log'] = pton_hist['Log Rets'] * mov_av['Position 2'].shift(1)
mov_av['P2 CumRet'] = np.exp(mov_av['Position 2 Log'].loc['2020-02-19':].cumsum())-1

# Cumulative Returns for a Passive Buy & Hold Strategy
mov_av['Buy & Hold CumRet'] = np.exp(pton_hist['Log Rets'].loc['2020-02-19':].cumsum())-1


# Creating a dataframe to find the windows that yield the best returns
top10df = pd.DataFrame(pton_hist['Close Price'])
top10df = top10df.merge(pton_hist['Log Rets'], how = 'left', 
                                  left_index = True, right_index = True)

top10df.columns = ['Price', 'Log Rets']

# Designate Ranges- (starting point, ending point, step size)
sma_range = range(100,365,5)
ema_range = range(14,100,2)

results = pd.DataFrame(columns = ['EMA Length', 'SMA Length', 'Cum Rets'])

# Compute the cumulative returns for designated range of EMA and SMA windows
# & sort them in decending order, displaying the top 10 strategies.
for ema1,sma1 in product(ema_range, sma_range):
    sma = pton_hist['Close Price'].rolling(window = sma1).mean()
    mod_price = pton_hist['Close Price'].copy()
    mod_price.iloc[0:ema1] = sma[0:ema1]
    ema = mod_price.ewm(span = ema1, adjust = False).mean()
    top10df['Position'] = np.where(ema > sma, 1, -1)
    top10df['pos_log'] = top10df['Log Rets'] * top10df['Position'].shift(1)
    cum_rets = np.exp(top10df['pos_log'].iloc[sma1:].sum())-1
    data = {'EMA Length': ema1, 'SMA Length': sma1, 'Cum Rets': cum_rets}
    results = results.append(data, ignore_index = True)
    
# Create a dataframe for the top 10 strategies    
top_strats = results.sort_values(by = ['Cum Rets'], ascending = False).head(10)

# Outputs
print("")
print('BACKTESTING RESULTS')
print("")
print(results.sort_values(by = ['Cum Rets'], ascending = False).head(10))
print("")
print('The cumulative return of a passive buy-and-hold strategy is'
      ,mov_av['Buy & Hold CumRet'][-1])
print("")
print('The cumulative return of the P1 long-short strategy is'
      ,mov_av['P1 CumRet'][-1])
print("")
print('The cumulative return of the P2 long-short strategy is'
      ,mov_av['P2 CumRet'][-1])

print("")
print("_________________________________________________________________")
print("")

ema_ideal = top_strats['EMA Length'].iloc[0].astype(int)
sma_ideal = top_strats['SMA Length'].iloc[0].astype(int)

# Create a dataframe for the ideal strategy
ideal_strat = pd.DataFrame(columns = [f'SMA{sma_ideal}'])

# Ideal Simple Moving Average
ideal_strat[f'SMA{sma_ideal}'] = pton_hist['Close Price'].rolling(
    window = sma_ideal).mean()

# Create new SMA to modify price data
sma_new = pton_hist['Close Price'].rolling(
    window = ema_ideal).mean()
mod_price = pton_hist['Close Price'].copy()
mod_price.iloc[0:ema_ideal] = sma_new[0:ema_ideal]

# Ideal Exponential Moving Average
ema_window = ema_ideal
ideal_strat[f'EMA{ema_ideal}'] = mod_price.ewm(span = ema_window, adjust = False).mean()

# Trading positions based on ideal EMA and ideal SMA
ideal_strat['Pos'] = np.where(ideal_strat[f'EMA{ema_ideal}'] > ideal_strat[
    f'SMA{sma_ideal}'], 1, -1)

# Merge the price with ideal_strat
ideal_strat = ideal_strat.merge(mov_av['Price'], how = 'left', 
                                  left_index = True, right_index = True)

# Calculate cumulative returns of the ideal strategy
ideal_logs = pton_hist['Log Rets'] * ideal_strat['Pos'].shift(1)
ideal_strat['CumRet'] = np.exp(ideal_logs.iloc[sma_ideal:].cumsum())-1

# Plot of Annual Time Series of ideal SMA & ideal EMA
ideal_strat.plot(y = [f'SMA{sma_ideal}', f'EMA{ema_ideal}', 'Price'])
plt.title(f'SMA{sma_ideal} & EMA{ema_ideal}')
plt.show()

# The terminal outputs whether we should take a long or short position
if ideal_strat['Pos'][-1] == 1 and ideal_strat['Pos'][-2] == -1:
    print('ESTABLISH A LONG POSITION')
elif ideal_strat['Pos'][-1] == 1 and ideal_strat['Pos'][-2] == 1:
    print('MAINTAIN LONG POSITION')
elif ideal_strat['Pos'][-1] == -1 and ideal_strat['Pos'][-2] == 1:
    print('ESTABLISH A SHORT POSITION')
else:
    print('MAINTAIN SHORT POSITION')
