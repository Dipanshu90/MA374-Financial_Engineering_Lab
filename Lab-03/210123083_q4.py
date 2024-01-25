import numpy as np
import matplotlib.pyplot as plt
import time
from functools import reduce
import operator as op

S0, K, T, r, sigma = 100, 100, 1, 0.08, 0.3
M1_array = [5, 10]
M2_array = [5, 10, 25, 50]
M3_array = [5, 10, 25, 50, 75, 100]

def no_arbitrage(u, d, r, t):
    if d < np.exp(r*t) and np.exp(r*t) < u:
        return True
    else:
        return False

def nCr(n, r):
    r = min(r, n-r)
    num = reduce(op.mul, range(n, n-r, -1), 1)
    den = reduce(op.mul, range(1, r+1), 1)
    return num // den  

def Option_price(i, S0, u, d, M):
    path = format(i, 'b').zfill(M)

    for idx in path:
        if idx == '1':
            S0 *= d
        else:
            S0 *= u
    
    return S0


def Binomial_Model_unoptimised(S0, K, T, M, r, sigma, display):
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
        price_array[M][i] = max(req_price - K, 0)
    
    for j in range(M - 1, -1, -1):
        for i in range(0, int(2**j)):
            price_array[j][i] = (p*price_array[j + 1][2*i] + (1 - p)*price_array[j + 1][2*i + 1]) / R

    if display == 1: 
        print(f"Initial Price of European Call Option \t= {price_array[0][0]}")
        print(f"Execution Time \t\t\t\t= {time.time() - curr_time} sec\n")
        
    return price_array[0][0]

def Binomial_Model_better(S0, K, T, M, r, sigma, display):
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

    price_array = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
        price_array[M][i] = max(0, S0*(u**(M - i))*(d**i) - K)
    
    for j in range(M - 1, -1, -1):
        for i in range(0, j+1):
            price_array[j][i] = (p*price_array[j + 1][i] + (1 - p)*price_array[j + 1][i + 1]) / R

    if display == 1: 
        print(f"Initial Price of European Call Option \t= {price_array[0][0]}")
        print(f"Execution Time \t\t\t\t= {time.time() - curr_time} sec\n")
    
    if display == 2:
        for i in range(len(price_array)):
            print(f"At t = {i}")
            for j in range(i+1):
                print(f"Serial no. {j+1}:-\tOption Price = {price_array[i][j]}")
            print()

    return price_array[0][0]

def Binomial_Model_best(S0, K, T, M, r, sigma, display):
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

    price = 0

    for j in range(0, M + 1):
        price += nCr(M, j) * (p**j) * ((1-p)**(M-j)) * max(S0 * (u**j) * (d**(M-j)) - K, 0)
  
    price /= (R**M)

    if display == 1: 
        print(f"Initial Price of European Call Option \t= {price}")
        print(f"Execution Time \t\t\t\t= {time.time() - curr_time} sec\n")

    return price

def plot_default(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.title(title)
    plt.show()

def main():
    print("************************  Q4(a)  ************************\n")
    prices_1, prices_2, prices_3 = [], [], []

    print("######################## Unoptimised Binomial Model ########################")
    for m in M1_array:
        prices_1.append(Binomial_Model_unoptimised(S0, K, T, m, r, sigma, display = 1))

    print("######################## Better Binomial Model ########################")
    for m in M2_array:
        prices_2.append(Binomial_Model_better(S0, K, T, m, r, sigma, display = 1))

    print("######################## Best Binomial Model ########################")
    for m in M3_array:
        prices_3.append(Binomial_Model_best(S0, K, T, m, r, sigma, display = 1))

    print("\n\n************************  Q4(b)  ************************")
    plot_default(M2_array, prices_2, "M", "Option Prices at t=0", "Initial Option Prices vs M")

    print("Calculating prices for M=1 to M=100 and comparing through plot.")

    M = [i for i in range(1,101)]
    prices_array = []
    for m in M:
        prices_array.append(Binomial_Model_best(S0, K, T, m, r, sigma, 0))
    
    plot_default(M, prices_array, "M", "Option Prices at t=0", "Initial Option Prices vs M(for M = 1 to 100)")

    print("\n\n************************  Q4(c)  ************************")
    Binomial_Model_better(S0, K, T, 5, r, sigma, display = 2)


if __name__=="__main__":
  main()