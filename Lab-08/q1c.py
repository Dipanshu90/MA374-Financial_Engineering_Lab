import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    df_reduced = df_monthly.iloc[start_idx:]
    df_reduced.reset_index(inplace=True, drop=True)
    idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
    df_reduced = df.iloc[idx_list[0]:]

    data = df_reduced.set_index('Date')
    # data = data.pct_change()
    data = data.pct_change(fill_method=None).dropna()
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

    d1 = (np.log(x/K) + (r + 0.5 * sigma * sigma)
          * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = (np.log(x/K) + (r - 0.5 * sigma * sigma)
          * (T - t)) / (sigma * np.sqrt(T - t))

    call_price = x * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    put_price = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - x * norm.cdf(-d1)

    return call_price, put_price


def price_computation(stocks_type):
    filename = ''
    if stocks_type == 'BSE':
        filename = './bsedata1.csv'
    else:
        filename = './nsedata1.csv'

    df = pd.read_csv(filename)
    # df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    stocks_name = df.columns[1:]
    r = 0.05
    t = 0
    T = 6/12
    sigma_list = []
    time_period = range(1, 61)

    for delta_t in range(1, 61):
        sigma_list.append(get_historical_volatility(stocks_type, delta_t))

    for idx1 in range(len(stocks_name)):
        print(
            "***********************  For stocks - {} ***********************".format(stocks_name[idx1]))
        plt.rcParams["figure.figsize"] = (20, 10)
        S0 = df.iloc[len(df) - 1][stocks_name[idx1]]
        call_prices, put_prices = np.zeros((11, 60)), np.zeros((11, 60))
        historical_volatility = []

        for idx2 in range(60):
            sigma = sigma_list[idx2][idx1]
            historical_volatility.append(sigma)
            A = [0.5, 1.0, 1.5]

            for idx3 in range(len(A)):
                K = A[idx3] * S0
                call, put = BSM_model(S0, t, T, K, r, sigma)
                call_prices[idx3][idx2] = call
                put_prices[idx3][idx2] = put

        for i in range(len(A)):
            ax = plt.subplot(1, 3, i + 1)
            plt.plot(time_period, call_prices[i])
            plt.xlabel("Length of time period (in months)")
            plt.ylabel("European Call Option Price")
            ax.set_title(
                "Call Option for {} with K = {}*S0".format(stocks_name[idx1], A[i]))
            if i == 2:
                plt.savefig('./Data/' + stocks_name[idx1] + '_Call.jpg')
                plt.close()

        for i in range(len(A)):
            ax = plt.subplot(1, 3, i + 1)
            plt.plot(time_period, put_prices[i])
            plt.xlabel("Length of time period (in months)")
            plt.ylabel("European Put Option Price")
            ax.set_title(
                "Put Option for {} with K = {}*S0".format(stocks_name[idx1], A[i]))
            if i == 2:
                plt.savefig('./Data/' + stocks_name[idx1] + '_Put.jpg')
                plt.close()

        plt.plot(time_period, historical_volatility)
        plt.xlabel("Length of time period (in months)")
        plt.ylabel("Volatility")
        plt.title("Historical Volatility vs time period for {}".format(
            stocks_name[idx1]))
        plt.savefig('./Data/' + stocks_name[idx1] + '_volatility.jpg')
        plt.close()


price_computation('NSE')
price_computation('BSE')
