import numpy as np
import matplotlib.pyplot as plt

def plot(prices, option, step):
    plt.plot(prices["M"], prices[f"{option}"])
    plt.xlabel('M')
    plt.ylabel(f'Initial {option} Prices')
    plt.title(f"{option} with step size {step}")
    plt.show()

def step_1(M_vals_1, S_0, K, T, r, sigma, count_1, C_prices_1, P_prices_1):
    print("M = 1 to 250 taking step size = 1.")
    for M in M_vals_1:
        del_T = T / M
        u = np.exp(sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        d = np.exp(-sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        p = (np.exp(r * del_T) - d) / (u - d)
        q = 1 - p

        if 0 < d and d < np.exp(r * del_T) and np.exp(r * del_T) < u:
            count_1 += 1
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

        C_prices_1.append(Call_list[0])
        P_prices_1.append(Put_list[0])

    prices_1 = {"M": M_vals_1,
            "Call": C_prices_1,
            "Put": P_prices_1}

    if count_1 == len(M_vals_1):
        print("No arbitrage condition satisfied for all configurations.\n")
    else:
        print("No-arbitrage condition failed for the above mentioned configurations.\n")

    print(f"Call option price:  {prices_1['Call'][-1]}")
    print(f"Put option price: {prices_1['Put'][-1]}\n")

    return prices_1

def step_5(M_vals_5, S_0, K, T, r, sigma, count_5, C_prices_5, P_prices_5):
    print("M = 1 to 250 taking step size = 5.")

    for M in M_vals_5:
        del_T = T / M
        u = np.exp(sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        d = np.exp(-sigma * np.sqrt(del_T) + (r - 0.5 * np.square(sigma)) * del_T)
        p = (np.exp(r * del_T) - d) / (u - d)
        q = 1 - p

        if 0 < d and d < np.exp(r * del_T) and np.exp(r * del_T) < u:
            count_5 += 1
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

        C_prices_5.append(Call_list[0])
        P_prices_5.append(Put_list[0])

    prices_5 = {"M": M_vals_5,
                "Call": C_prices_5,
                "Put": P_prices_5}

    if count_5 == len(M_vals_5):
        print("No arbitrage condition satisfied for all configurations.\n")
    else:
        print("No-arbitrage condition failed for the above mentioned configurations.\n")

    print(f"Call option price:  {prices_5['Call'][-1]}")
    print(f"Put option price: {prices_5['Put'][-1]}")

    return prices_5 

def main():
    M_vals_1 = [i for i in range(1, 252, 1)]
    M_vals_5 = [i for i in range(1, 256, 5)]
    S_0, K, T, r, sigma = 100, 105, 5, 0.05, 0.4
    count_1, count_5 = 0, 0
    C_prices_1, P_prices_1 = list(), list()
    C_prices_5, P_prices_5 = list(), list()

    prices_1 = step_1(M_vals_1, S_0, K, T, r, sigma, count_1, C_prices_1, P_prices_1)

    print("-------------------------------------------------------------\n")

    prices_5 = step_5(M_vals_5, S_0, K, T, r, sigma, count_5, C_prices_5, P_prices_5)

    plot(prices_1, "Call", 1)
    plot(prices_1, "Put", 1)
    plot(prices_5, "Call", 5)
    plot(prices_5, "Put", 5)
    return 0

if __name__ == "__main__":
    main()