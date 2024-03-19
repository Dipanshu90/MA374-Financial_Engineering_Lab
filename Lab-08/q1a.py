import numpy as np
import pandas as pd
from tabulate import tabulate


def get_historical_volatility(stocks_type, time_period):
    filename, stocks_name = '', []
    if stocks_type == 'BSE':
        filename = './bsedata1.csv'
    else:
        filename = './nsedata1.csv'

    df = pd.read_csv(filename)
    # df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)

    start_idx = 60 - time_period
    df_reduced = df_monthly.iloc[start_idx:]
    df_reduced.reset_index(inplace=True, drop=True)
    idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
    df_reduced = df.iloc[idx_list[0]:]

    data = df_reduced.set_index('Date')
    data = data.pct_change().dropna()
    # print(data)

    stocks_name = data.columns

    volatility = []
    for sname in stocks_name:
        if sname == 'Date':
            continue
        returns = data[sname]
        x = returns.to_list()
        mean = np.nanmean(np.array(x))
        std = np.nanstd(np.array(x))

        volatility.append(std * np.sqrt(252))

    table = []
    for i in range(len(volatility)):
        table.append([i + 1, stocks_name[i], volatility[i]])

    print(tabulate(table, headers=[
          'SI No', 'Stocks Name', 'Historical Volatility']))


print("**********  Historical Volatility of last month for BSE  **********")
get_historical_volatility('BSE', 1)

print("\n\n**********  Historical Volatility of last month for NSE  **********")
get_historical_volatility('NSE', 1)
