import numpy as np
import matplotlib.pyplot as plt
import time

S0, T, r, sigma = 100, 1, 0.08, 0.3
M_array = [5,10,25]

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


def lookback_option_Binomial_Model(S0, T, M, r, sigma, display):
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
        X = []
        for j in range(int(2**i)):
            X.append(0)
        price_array.append(X)
        
    for i in range(int(2**M)):
        req_price = Option_price(i, S0, u, d, M)
        price_array[M][i] = max(req_price, 0)
    
    for j in range(M - 1, -1, -1):
        for i in range(0, int(2**j)):
            price_array[j][i] = (p*price_array[j + 1][2*i] + (1 - p)*price_array[j + 1][2*i + 1]) / R

    if display == 1: 
        print(f"Initial Price of Lookback Option \t= {price_array[0][0]}")
        print(f"Execution Time \t\t\t\t= {time.time() - curr_time} sec\n")

    if display == 2:
        for i in range(len(price_array)):
            print(f"At t = {i}")
            for j in range(len(price_array[i])):
                print(f"Serial no. {j+1}:-\tOption Price = {price_array[i][j]}")
            print()
        
    return price_array[0][0]

def plot_default(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.title(title)
    plt.show()

def main():
    print("************************  Q2(a)  ************************")
    prices = []
    for m in M_array:
        prices.append(lookback_option_Binomial_Model(S0, T, m, r, sigma, display = 1))

    # The execution of above for loop will take around 2 mins for M = 25 and it takes too much time for M=50.
    # So I didn't include M-50 in the above defined M_array, if you want to check the result for M=50 too, please include it in the array.

    print("\n\n************************  Q2(b)  ************************")
    plot_default(M_array, prices, "M", "Option Prices at t=0", "Initial Option Prices vs M")

    print("Calculating prices for M=1 to M=20 and comparing through plot.")

    M = [i for i in range(1,21)]
    prices_array = []
    for m in M:
        prices_array.append(lookback_option_Binomial_Model(S0, T, m, r, sigma, 0))
    
    plot_default(M, prices_array, "M", "Option Prices at t=0", "Initial Option Prices vs M(for M = 1 to 20)")

    print("\n\n************************  Q2(c)  ************************")
    lookback_option_Binomial_Model(S0, T, 5, r, sigma, display = 2)


if __name__=="__main__":
  main()