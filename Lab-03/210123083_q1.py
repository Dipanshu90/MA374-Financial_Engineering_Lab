import numpy as np
import matplotlib.pyplot as plt

S0, K, T, r, sigma = 100, 100, 1, 0.08, 0.3
M=100

def no_arbitrage(u, d, r, t):
    if d < np.exp(r*t) and np.exp(r*t) < u:
        return True
    else:
        return False

def Binomial_Model_American(S0, K, T, M, r, sigma):
    u, d = 0, 0
    t = T/M
    u = np.exp(sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = np.exp(-sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)  

    R = np.exp(r*t)
    p = (R - d)/(u - d)

    if no_arbitrage(u, d, r, t) == False:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0, 0

    call_price_array = [[0 for i in range(M + 1)] for j in range(M + 1)]
    put_price_array = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
        call_price_array[M][i] = max(0, S0*(u**(M - i))*(d**i) - K)
        put_price_array[M][i] = max(0, K - S0*(u**(M - i))*(d**i))
    
    for i in range(M - 1, -1, -1):
        for j in range(0, len(call_price_array)-1):
            S_price = S0 * (u ** (i - j)) * (d ** j)
            call_price_array[i][j] = max((p*call_price_array[i + 1][j] + (1 - p)*call_price_array[i + 1][j + 1]) / R, S_price - K, 0)
            put_price_array[i][j] = max((p*put_price_array[i + 1][j] + (1 - p)*put_price_array[i + 1][j + 1]) / R, K - S_price, 0)

    return call_price_array[0][0], put_price_array[0][0]

def plot_default(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.title(title)
    plt.show()

def plot_S0():
    S =  np.linspace(20, 200, 50)

    call_prices = []
    put_prices = []
    for s in S:
        c, p = Binomial_Model_American(s, K, T, M, r, sigma)
        call_prices.append(c)
        put_prices.append(p)

    plot_default(S, call_prices, "S0", "Prices of Call option at t = 0", "Initial Call Option Price vs S0")
    plot_default(S, put_prices, "S0", "Prices of Put option at t = 0", "Initial Put Option Price vs S0")


def plot_K():
    K =  np.linspace(20, 200, 50)

    call_prices = []
    put_prices = []
    for k in K:
        c, p = Binomial_Model_American(S0, k, T, M, r, sigma)
        call_prices.append(c)
        put_prices.append(p)

    plot_default(K, call_prices, "K", "Prices of Call option at t = 0", "Initial Call Option Price vs K")
    plot_default(K, put_prices, "K", "Prices of Put option at t = 0", "Initial Put Option Price vs K")


def plot_r():
    r_list =  np.linspace(0, 1, 100)

    call_prices = []
    put_prices = []
    for rate in r_list:
        c, p = Binomial_Model_American(S0, K, T, M, rate, sigma)
        call_prices.append(c)
        put_prices.append(p)

    plot_default(r_list, call_prices, "r", "Prices of Call option at t = 0", "Initial Call Option Price vs r")
    plot_default(r_list, put_prices, "r", "Prices of Put option at t = 0", "Initial Put Option Price vs r")


def plot_sigma():
    sigma_list =  np.linspace(0.01, 1, 100)

    call_prices = []
    put_prices = []
    for sg in sigma_list:
        c, p = Binomial_Model_American(S0, K, T, M, r, sg)
        call_prices.append(c)
        put_prices.append(p)

    plot_default(sigma_list, call_prices, "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma")
    plot_default(sigma_list, put_prices, "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma")


def plot_M():
    M_list =  [i for i in range(50, 201)]
    K_list = [95, 100, 105]

    for k in K_list:
        call_prices = []
        put_prices = []
        for m in M_list:
            c, p = Binomial_Model_American(S0, k, T, m, r, sigma)
            call_prices.append(c)
            put_prices.append(p)

        plot_default(M_list, call_prices, "M", "Prices of Call option at t = 0", "Initial Call Option Price vs M for K = " + str(k))
        plot_default(M_list, put_prices, "M", "Prices of Put option at t = 0", "Initial Put Option Price vs M for K = " + str(k))

def main():
    call_price, put_price = Binomial_Model_American(S0, K, T, M, r, sigma)
    print(f"American Call Option Price = {call_price}")
    print(f"American Put Option Price = {put_price}")
    plot_S0()
    plot_K()
    plot_r()
    plot_sigma()
    plot_M()

if __name__=="__main__":
  main()