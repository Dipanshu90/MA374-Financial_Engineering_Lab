import numpy as np
import matplotlib.pyplot as plt
import time

S0, T, r, sigma = 100, 1, 0.08, 0.3
M_array = [5,10,25,50]

def no_arbitrage(u, d, r, t):
    if d < np.exp(r*t) and np.exp(r*t) < u:
        return True
    else:
        return False


def Option_price(i, S0, u, d, M):
    path = format(i, 'b').zfill(M)
    curr_max = S0

    for idx in path:
        if idx == '1':
            S0 *= d
        else:
            S0 *= u

    curr_max = max(curr_max, S0)

    return curr_max - S0

def optimised_price_calculator(idx, u, d, p, R, M, stock_price, curr_max, option_prices):
    if idx == M + 1 or (stock_price, curr_max) in option_prices[idx]:
        return

    optimised_price_calculator(idx + 1, u, d, p, R, M, stock_price*u, max(stock_price*u, curr_max), option_prices)
    optimised_price_calculator(idx + 1, u, d, p, R, M, stock_price*d, max(stock_price*d, curr_max), option_prices)

    if idx == M:
        option_prices[M][(stock_price, curr_max)] = max(curr_max - stock_price, 0)
    else:
        option_prices[idx][(stock_price, curr_max)] = (p*option_prices[idx + 1][ (u * stock_price, max(u * stock_price, curr_max)) ] + (1 - p)*option_prices[idx + 1][ (d * stock_price, curr_max) ]) / R

def lookback_option_Markov_based(S0, T, M, r, sigma, display):
    if display == 1: 
        print(f"\n\n-----------  Executing for M = {M}  -----------\n")
    curr_time = time.time()
    
    u, d = 0, 0
    t = T/M
    u = np.exp(sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = np.exp(-sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)  

    R = np.exp(r*t)
    p = (R - d)/(u - d)

    if no_arbitrage(u, d, r, t) == False:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0, 0

    price_array = []
    for i in range(0, M + 1):
        price_array.append(dict())
    
    optimised_price_calculator(0, u, d, p, R, M, S0, S0, price_array)

    if display == 1: 
        print(f"Initial Price of Lookback Option \t= {price_array[0][ (S0,S0) ]}")
        print(f"Execution Time \t\t\t\t= {time.time() - curr_time} sec\n")

    if display == 2:
        for i in range(len(price_array)):
            print(f"At t = {i}")
            j=1
            for key, price in price_array[i].items():
                print(f"Serial no. {j}:-\tOption Price = {price}")
                j = j+1
            print()
        
    return price_array[0][ (S0, S0) ]

def plot_default(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.title(title)
    plt.show()

def main():
    print("************************  Q3(a)  ************************")
    prices = []
    for m in M_array:
        prices.append(lookback_option_Markov_based(S0, T, m, r, sigma, display = 1))

    print("\n\n************************  Q3(b)  ************************")
    plot_default(M_array, prices, "M", "Option Prices at t=0", "Initial Option Prices vs M")

    print("Calculating prices for M=1 to M=40 and comparing through plot.")

    M = [i for i in range(1,41)]
    prices_array = []
    for m in M:
        prices_array.append(lookback_option_Markov_based(S0, T, m, r, sigma, 0))
    
    plot_default(M, prices_array, "M", "Option Prices at t=0", "Initial Option Prices vs M(for M = 1 to 40)")

    print("\n\n************************  Q3(c)  ************************")
    lookback_option_Markov_based(S0, T, 5, r, sigma, display = 2)


if __name__=="__main__":
  main()