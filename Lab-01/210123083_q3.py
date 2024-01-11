import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

S_0, K, T, r, sigma = 100, 105, 5, 0.05, 0.4
M = 20
times = np.array([0, 0.50, 1, 1.50, 3, 4.5])
C_prices, P_prices = {}, {}

def main():
    del_T = T / M
    itrs = times / del_T
    u = np.exp(sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
    d = np.exp(-sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
    p = (np.exp(r * del_T) - d) / (u - d)
    q = 1 - p

    if 0 < d and d < np.exp(r * del_T) and np.exp(r * del_T) < u:
        print(f"No arbitrage condition is satisfied for M = {M}.")
    else:
        print(f"No arbitrage condition is not satisfied for M = {M}.")
        return 0

    Call_list, Put_list = list(), list()

    for i in range(M + 1):
        S_t = S_0 * (u ** (M - i)) * (d ** i)
        price_call = max(0, S_t - K)
        price_put = max(0, K - S_t)
        Call_list.append(price_call)
        Put_list.append(price_put)

    for i in range(M-1, -1, -1):
        New_call_list, New_put_list = list(), list()

        for j in range(len(Call_list) - 1):
            New_call_list.append(np.exp(-r * del_T) * (Call_list[j] * p + Call_list[j + 1] * q))

            New_put_list.append(np.exp(-r * del_T) * (Put_list[j] * p + Put_list[j + 1] * q))

        Put_list, Call_list = New_put_list, New_call_list

        if i in itrs:
            C_prices[str(i)], P_prices[str(i)] = Call_list, Put_list

    for i, j in enumerate(C_prices):
        print(f"T = {int(j) * del_T}")
        print(pd.DataFrame({"Call Price": C_prices[j], "Put Price": P_prices[j]}))
        print()

    return 0

if __name__ == "__main__":
    main()