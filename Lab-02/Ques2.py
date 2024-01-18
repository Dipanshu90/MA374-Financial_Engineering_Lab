import numpy as np
import matplotlib.pyplot as plt

S0, K, T, M, r, sigma = 100, 100, 1, 10, 0.08, 0.30

def no_arbitrage(u, d, r, t):
  if d < np.exp(r*t) and np.exp(r*t) < u:
    return False
  else:
    return True


def Option_price(i, S0, u, d, M):
  path = format(i, 'b').zfill(M)
  path = path[::-1]
  sum = S0
  for idx in path:
    if idx == '1':
      S0 *= d
    else:
      S0 *= u
    sum += S0
  
  return sum/(1 + M)


def Binomial_model(S0, K, T, M, r, sigma, set):
  u, d = 0, 0
  t = T/M

  if set == 1:
    u = np.exp(sigma*np.sqrt(t))
    d = np.exp(-sigma*np.sqrt(t)) 
  else:
    u = np.exp(sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = np.exp(-sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)

  R = np.exp(r*t)
  p = (R - d)/(u - d)
  cond = no_arbitrage(u, d, r, t)

  if cond:
    return 0, 0

  C, P = [], []
  for i in range(0, M + 1):
    D, E = [], []
    for j in range(int(2 ** i)):
      D.append(0)
      E.append(0)
    C.append(D)
    P.append(E)

  for i in range(int(2 ** M)):
    avg_price = Option_price(i, S0, u, d, M)
    C[M][i] = max(avg_price - K, 0)
    P[M][i] = max(K - avg_price, 0)
  
  for j in range(M - 1, -1, -1):
    for i in range(0, int(2 ** j)):
      C[j][i] = (p*C[j + 1][2*i] + (1 - p)*C[j + 1][2*i + 1]) / R
      P[j][i] = (p*P[j + 1][2*i] + (1 - p)*P[j + 1][2*i + 1]) / R

  return C[0][0], P[0][0]

def plot_fixed(x, y, x_label, y_label, title):
  plt.plot(x, y)
  plt.xlabel(x_label)
  plt.ylabel(y_label) 
  plt.title(title)
  plt.show()

def plot_S0():
  S =  np.linspace(20, 200, 100)

  for set_num in range(2, 3):
    call_prices = []
    put_prices = []
    for s in S:
      c, p = Binomial_model(s, K, T, M, r, sigma, set = set_num)
      call_prices.append(c)
      put_prices.append(p)

    plot_fixed(S, call_prices, "S0", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 for the set = " + str(set_num))
    plot_fixed(S, put_prices, "S0", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 for the set = " + str(set_num))


def plot_K():
  K =  np.linspace(20, 200, 100)

  for set_num in range(2, 3):
    call_prices = []
    put_prices = []
    for k in K:
      c, p = Binomial_model(S0, k, T, M, r, sigma, set = set_num)
      call_prices.append(c)
      put_prices.append(p)

    plot_fixed(K, call_prices, "K", "Prices of Call option at t = 0", "Initial Call Option Price vs K for the set = " + str(set_num))
    plot_fixed(K, put_prices, "K", "Prices of Put option at t = 0", "Initial Put Option Price vs K for the set = " + str(set_num))


def plot_r():
  r_list =  np.linspace(0, 1, 100)

  for set_num in range(2, 3):
    call_prices = []
    put_prices = []
    for rate in r_list:
      c, p = Binomial_model(S0, K, T, M, rate, sigma, set = set_num)
      call_prices.append(c)
      put_prices.append(p)

    plot_fixed(r_list, call_prices, "r", "Prices of Call option at t = 0", "Initial Call Option Price vs r for the set = " + str(set_num))
    plot_fixed(r_list, put_prices, "r", "Prices of Put option at t = 0", "Initial Put Option Price vs r for the set = " + str(set_num))


def plot_sigma():
  sigma_list =  np.linspace(0.01, 1, 100)

  for set_num in range(2, 3):
    call_prices = []
    put_prices = []
    for sg in sigma_list:
      c, p = Binomial_model(S0, K, T, M, r, sg, set = set_num)
      call_prices.append(c)
      put_prices.append(p)

    plot_fixed(sigma_list, call_prices, "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma for the set = " + str(set_num))
    plot_fixed(sigma_list, put_prices, "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma for the set = " + str(set_num))


def plot_M():
  M_list =  [i for i in range(8, 15)]
  K_list = [95, 100, 105]

  for k in K_list:
    for set_num in range(2, 3):
      call_prices = []
      put_prices = []
      for m in M_list:
        c, p = Binomial_model(S0, k, T, m, r, sigma, set = set_num)
        call_prices.append(c)
        put_prices.append(p)

      plot_fixed(M_list, call_prices, "M", "Prices of Call option at t = 0", "Initial Call Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))
      plot_fixed(M_list, put_prices, "M", "Prices of Put option at t = 0", "Initial Put Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))

def main():
  c, p = Binomial_model(S0, K, T, M, r, sigma, set = 2)
  print(f"Call Option = {c}")
  print(f"Put Option = {p}\n\n")
  plot_S0()
  plot_K()
  plot_r()
  plot_sigma()
  plot_M()


if __name__=="__main__":
  main()
