import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

def get_historical_volatility(stocks_type, time_period):
    filename = ''
    if stocks_type == 'BSE':
        filename = './bsedata1.csv'
    else:
        filename = './nsedata1.csv'
  
    df = pd.read_csv(filename)
    # df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)

    start_idx = 60 - time_period
    df_reduced = df_monthly.iloc[start_idx :]
    df_reduced.reset_index(inplace = True, drop = True) 
    idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
    df_reduced = df.iloc[idx_list[0] :]

    data = df_reduced.set_index('Date')
    data = data.pct_change().dropna()
    stocks_name = df_reduced.columns[1:]

    volatility = []
    for sname in stocks_name:
        returns = data[sname]
        x = returns.to_list()
        mean = np.nanmean(np.array(x))
        std = np.nanstd(np.array(x))
        
        volatility.append(std * np.sqrt(252))
    
    table = []
    for i in range(len(volatility)):
        table.append([i + 1, stocks_name[i], volatility[i]])
    
    return volatility


def BSM_model(x, t, T, K, r, sigma):
    if t == T:
        return max(0, x - K), max(0, K - x)

    d1 = ( np.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * np.sqrt(T - t) )
    d2 = ( np.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * np.sqrt(T - t) )

    call_price = x * norm.cdf(d1) - K * np.exp( -r * (T - t) ) * norm.cdf(d2)
    put_price = K * np.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)

    return call_price, put_price


def price_computation(stocks_type):
    filename = ''
    if stocks_type == 'BSE':
        filename = './bsedata1.csv'
    else:
        filename = './nsedata1.csv'
    
    df = pd.read_csv(filename)
    df = df.interpolate(method ='linear', limit_direction ='forward')
    r = 0.05
    t = 0
    T = 6/12
    sigma_list = get_historical_volatility(stocks_type, 1)
    stocks_name = df.columns[1:]
    
    for idx1 in range(len(stocks_name)):
        print("\n\n***********************  For stocks - {} ***********************".format(stocks_name[idx1]))
        sigma = sigma_list[idx1]
        print("Historical volatility for last 1 month\t=", sigma)
        S0 = df.iloc[len(df) - 1][stocks_name[idx1]]
        table = []

        for idx2 in [0.5, 1.0, 1.5]:
            K = S0 * round(idx2, 2)
            call, put = BSM_model(S0, 0, T, K, r, sigma)
            # table.append([str(round(idx2, 2)) + "*S0", call, put])
            table.append(["{:.2f}*S0".format(round(idx2, 2)), "{:.2f}".format(call), "{:.2f}".format(put)])
    
        print(tabulate(table, headers = ["Strike price (K)", "Call Option", "Put Option"]))

price_computation('BSE')
price_computation('NSE')