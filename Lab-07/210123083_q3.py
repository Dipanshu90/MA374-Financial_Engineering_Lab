from math import sqrt, log, exp
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def calEurCallPutPrices(T, K, S, r, sigma, t):
    if(T == t):
        putp = max(K-S, 0)
        callp = max(S-K, 0)
        return [callp, putp]

    d1 = (log(S/K)+(r+(sigma**2/2))*(T-t))/(sigma*sqrt(T-t))
    d2 = d1-sigma*sqrt(T-t)

    putp = K*exp(-r*(T-t))*norm.cdf(-d2) - S*norm.cdf(-d1)
    callp = S*norm.cdf(d1)-K*norm.cdf(d2)*exp(-r*(T-t))
    return [callp, putp]


def main():
    # Given data
    T, K, r, sig = 1, 1, 0.05, 0.6
    t = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    S = np.arange(0.05, 3.05, 0.05)

    # Fill 2-D arrays will zeros for storing prices
    callPrices = np.zeros(shape=(t.shape[0], S.shape[0]))
    putPrices = np.zeros(shape=(t.shape[0], S.shape[0]))

    for i in range(t.shape[0]):
        for j in range(S.shape[0]):
            [callPrices[i][j], putPrices[i][j]] = calEurCallPutPrices(T, K, S[j], r, sig, t[i])

    callPricesT = np.matrix.transpose(callPrices)
    putPricesT = np.matrix.transpose(putPrices)

    tx, Sy = np.meshgrid(t, S)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(tx, Sy, callPricesT, cmap='viridis')
    ax.set_title('Surface Plot of C(t,x) vs t and x')
    ax.set_xlabel('Stock Prices')
    ax.set_ylabel('Time')
    ax.set_zlabel('Price of European Call Option')
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(tx, Sy, putPricesT, cmap='viridis')
    ax.set_title('Surface Plot of P(t,x) vs t and x')
    ax.set_xlabel('Stock Prices')
    ax.set_ylabel('Time')
    ax.set_zlabel('Price of European Put Option')
    plt.show()

if __name__ == '__main__':
    main()