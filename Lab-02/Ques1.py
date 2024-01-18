import numpy as np
import matplotlib.pyplot as plt

S0, K, T, M, r, sigma = 100, 100, 1, 100, 0.08, 0.30

def no_arbitrage(u, d, r, t):
  if d < np.exp(r*t) and np.exp(r*t) < u:
    return False
  else:
    return True


def Binomial_model(S0, K, T, M, r, sigma, set):
  t = T/M
  u,d = 0,0

  if set == 1:
    u = np.exp(sigma*np.sqrt(t))
    d = np.exp(-sigma*np.sqrt(t)) 
  else:
    u = np.exp(sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = np.exp(-sigma*np.sqrt(t) + (r - 0.5*sigma*sigma)*t)

  R = np.exp(r*t)
  p = (R - d)/(u - d);
  cond = no_arbitrage(u, d, r, t)

  if cond:
    return 0

  C = [[0 for i in range(M + 1)] for j in range(M + 1)]
  P = [[0 for i in range(M + 1)] for j in range(M + 1)]

  for i in range(0, M + 1):
    C[M][i] = max(0, S0*(u ** (M-i))*(d ** i) - K)
    P[M][i] = max(0, K - S0*(u ** (M-i))*(d ** i))

  for j in range(M - 1, -1, -1):
    for i in range(0, j + 1):
      C[j][i] = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R
      P[j][i] = (p*P[j + 1][i] + (1 - p)*P[j + 1][i + 1]) / R

  return C[0][0], P[0][0]

def plot_fixed(x, y, x_label, y_label, title):
  plt.plot(x, y)
  plt.xlabel(x_label)
  plt.ylabel(y_label) 
  plt.title(title)
  plt.show()

def plot_fixed_3d(x, y, z, x_label, y_label, z_label, title):
  ax = plt.axes(projection='3d')
  ax.scatter3D(x, y, z, cmap='Greens')
  ax.set_xlabel(x_label) 
  ax.set_ylabel(y_label) 
  ax.set_zlabel(z_label)
  plt.title(title)
  plt.show()

def plot_S0():
  S =  np.linspace(20, 200, 100)

  for set_num in range(1, 3):
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

  for set_num in range(1, 3):
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

  for set_num in range(1, 3):
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

  for set_num in range(1, 3):
    call_prices = []
    put_prices = []
    for sg in sigma_list:
      c, p = Binomial_model(S0, K, T, M, r, sg, set = set_num)
      call_prices.append(c)
      put_prices.append(p)

    plot_fixed(sigma_list, call_prices, "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma for the set = " + str(set_num))
    plot_fixed(sigma_list, put_prices, "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma for the set = " + str(set_num))


def plot_M():
  M_list =  [i for i in range(50, 200)]
  K_list = [95, 100, 105]

  for k in K_list:
    for set_num in range(1, 3):
      call_prices = []
      put_prices = []
      for m in M_list:
        c, p = Binomial_model(S0, k, T, m, r, sigma, set = set_num)
        call_prices.append(c)
        put_prices.append(p)

      plot_fixed(M_list, call_prices, "M", "Prices of Call option at t = 0", "Initial Call Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))
      plot_fixed(M_list, put_prices, "M", "Prices of Put option at t = 0", "Initial Put Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))


def plot_S0_K():
  S =  np.linspace(20, 200, 50)
  K =  np.linspace(20, 200, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for s in S:
      for k in K:
        c, p = Binomial_model(s, k, T, M, r, sigma, set = set_num)
        x_axis.append(s)
        y_axis.append(k)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "S0", "K", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and K for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "S0", "K", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and K for the set = " + str(set_num))


def plot_S0_r():
  S =  np.linspace(20, 200, 50)
  r_list =  np.linspace(0, 1, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for s in S:
      for rate in r_list:
        c, p = Binomial_model(s, K, T, M, rate, sigma, set = set_num)
        x_axis.append(s)
        y_axis.append(rate)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "S0", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and r for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "S0", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and r for the set = " + str(set_num))


def plot_S0_sigma():
  S =  np.linspace(20, 200, 50)
  sigma_list =  np.linspace(0.01, 1, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for s in S:
      for sg in sigma_list:
        c, p = Binomial_model(s, K, T, M, r, sg, set = set_num)
        x_axis.append(s)
        y_axis.append(sg)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "S0", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and sigma for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "S0", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and sigma for the set = " + str(set_num))


def plot_S0_M():
  S =  np.linspace(20, 200, 50)
  M_list =  [i for i in range(50, 200)]

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for s in S:
      for m in M_list:
        c, p = Binomial_model(s, K, T, m, r, sigma, set = set_num)
        x_axis.append(s)
        y_axis.append(m)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "S0", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs S0 and M for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "S0", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs S0 and M for the set = " + str(set_num))


def plot_K_r():
  K =  np.linspace(20, 200, 50)
  r_list =  np.linspace(0, 1, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for k in K:
      for rate in r_list:
        c, p = Binomial_model(S0, k, T, M, rate, sigma, set = set_num)
        x_axis.append(k)
        y_axis.append(rate)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "K", "r", "Prices of Call option at t = 0", "Initial Call Option Price vs K and r for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "K", "r", "Prices of Put option at t = 0", "Initial Put Option Price vs K and r for the set = " + str(set_num))


def plot_K_sigma():
  K =  np.linspace(20, 200, 50)
  sigma_list =  np.linspace(0.01, 1, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for k in K:
      for sg in sigma_list:
        c, p = Binomial_model(S0, k, T, M, r, sg, set = set_num)
        x_axis.append(k)
        y_axis.append(sg)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "K", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs K and sigma for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "K", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs K and sigma for the set = " + str(set_num))


def plot_K_M():
  K =  np.linspace(20, 200, 50)
  M_list =  [i for i in range(50, 200, 2)]

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for k in K:
      for m in M_list:
        c, p = Binomial_model(S0, k, T, m, r, sigma, set = set_num)
        x_axis.append(k)
        y_axis.append(m)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "K", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs K and M for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "K", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs K and M for the set = " + str(set_num))


def plot_r_sigma():
  r_list =  np.linspace(0, 1, 50)
  sigma_list =  np.linspace(0.15, 1, 50)

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for rate in r_list:
      for sg in sigma_list:
        c, p = Binomial_model(S0, K, T, M, rate, sg, set = set_num)
        x_axis.append(rate)
        y_axis.append(sg)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "r", "sigma", "Prices of Call option at t = 0", "Initial Call Option Price vs r and sigma for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "r", "sigma", "Prices of Put option at t = 0", "Initial Put Option Price vs r and sigma for the set = " + str(set_num))


def plot_r_M():
  r_list =  np.linspace(0, 1, 50)
  M_list =  [i for i in range(50, 200, 2)]

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for rate in r_list:
      for m in M_list:
        c, p = Binomial_model(S0, K, T, m, rate, sigma, set = set_num)
        x_axis.append(rate)
        y_axis.append(m)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "r", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs r and M for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "r", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs r and M for the set = " + str(set_num))


def plot_sigma_M():
  sigma_list =  np.linspace(0.1, 1, 50)
  M_list =  [i for i in range(50, 200, 2)]

  for set_num in range(1, 3):
    call_prices, put_prices = [], []
    x_axis, y_axis = [], []
    for sg in sigma_list:
      for m in M_list:
        c, p = Binomial_model(S0, K, T, m, r, sg, set = set_num)
        x_axis.append(sg)
        y_axis.append(m)
        call_prices.append(c)
        put_prices.append(p)

    plot_fixed_3d(x_axis, y_axis, call_prices, "sigma", "M", "Prices of Call option at t = 0", "Initial Call Option Price vs sigma and M for the set = " + str(set_num))
    plot_fixed_3d(x_axis, y_axis, put_prices, "sigma", "M", "Prices of Put option at t = 0", "Initial Put Option Price vs sigma and M for the set = " + str(set_num))


def main():
  c,p = Binomial_model(S0, K, T, M, r, sigma, set = 1)
  print(f"Set = 1")
  print(f"Call Option = {c}")
  print(f"Put Option = {p}\n\n")

  c,p = Binomial_model(S0, K, T, M, r, sigma, set = 2)
  print(f"Set = 2")
  print(f"Call Option = {c}")
  print(f"Put Option = {p}\n\n")
  plot_S0()
  plot_K()
  plot_r()
  plot_sigma()
  plot_M()
  plot_S0_K()
  plot_S0_r()
  plot_S0_sigma()
  plot_S0_M()
  plot_K_r()
  plot_K_sigma()
  plot_K_M()
  plot_r_sigma()
  plot_r_M()
  plot_sigma_M()
  
  # Execution of all the above functions may take a lot of time. To run specific ones, uncomment the rest function calls


if __name__=="__main__":
  main()
