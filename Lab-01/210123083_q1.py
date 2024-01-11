import numpy as np
import pandas as pd

M_vals = [1, 5, 10, 20, 50, 100, 200, 400]
S_0, K, T, r, sigma = 100, 105, 5, 0.05, 0.4
C_prices = list()
P_prices = list()

def main():
    for M in M_vals:
        del_T = T / M
        u = np.exp(sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        d = np.exp(-sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        p = (np.exp(r * del_T) - d) / (u - d)
        q = 1 - p

        if 0 < d and d < np.exp(r * del_T) and np.exp(r * del_T) < u:
            print(f"No arbitrage condition is satisfied for M = {M}.")
        else:
            print(f"No arbitrage condition is not satisfied for M = {M}.")
            break
            
        Call_list, Put_list = list(), list()

        for i in range(M + 1):
            S_t = S_0 * (u ** (M - i)) * (d ** i)
            price_call = max(0, S_t - K)
            price_put = max(0, K - S_t)
            Call_list.append(price_call)
            Put_list.append(price_put)

        for i in range(M):
            New_call_list = list()
            New_put_list = list()

            for j in range(len(Call_list) - 1):
                C_new = np.exp(-r * del_T) * (Call_list[j] * p + Call_list[j + 1] * q)
                New_call_list.append(C_new)
                
                P_new = np.exp(-r * del_T) * (Put_list[j] * p + Put_list[j + 1] * q)
                New_put_list.append(P_new)
                
            Put_list, Call_list = New_put_list, New_call_list

        C_prices.append(Call_list[0])
        P_prices.append(Put_list[0])

    prices = {"M": M_vals,
            "Call price" : C_prices,
            "Put price": P_prices}
    print()
    print(pd.DataFrame(prices))

    return 0

if __name__ == "__main__":
    main()