import numpy as np
import pandas as pd

def find_market_portfolio(filename):
  df = pd.read_csv(filename)
  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
  df.sort_values(by=['Date'], inplace=True)
  df.set_index('Date', inplace=True)
  daily_returns = (df['Open'] - df['Price'])/df['Open']
  daily_returns = np.array(daily_returns)

  df = pd.DataFrame(np.transpose(daily_returns))
  M, sigma = np.mean(df, axis = 0) * len(df) / 5, df.std()
  
  mu_market = M[0]
  risk_market = sigma[0]

  return mu_market, risk_market



def execute_model(stocks_name, type, mu_market_index, risk_market_index, beta):
  daily_returns = []
  mu_rf = 0.05

  for i in range(len(stocks_name)):
    filename = './210123083_Data/' + type + '_Stocks/' + stocks_name[i] + '.csv'
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.sort_values(by=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    df = df.pct_change()
    daily_returns.append(df['Open'])

  daily_returns = np.array(daily_returns)
  df = pd.DataFrame(np.transpose(daily_returns), columns = stocks_name)
  M = np.mean(df, axis = 0) * len(df) / 5
  C = df.cov()
  
  print("\n\nStocks Name\t\t\t\tActual Return\t\t\tExpected Return\n")
  for i in range(len(M)):
    print(f"{stocks_name[i] : <20}\t\t\t{M[i] : ^10}\t\t{(beta[i] * (mu_market_index - mu_rf) + mu_rf) : >10}")


def compute_beta(stocks_name, main_filename, index_type):
  df = pd.read_csv(main_filename)
  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
  df.sort_values(by=['Date'], inplace=True)
  df.set_index('Date', inplace=True)
  daily_returns = (df['Open'] - df['Price'])/df['Open']

  daily_returns_stocks = []
    
  for i in range(len(stocks_name)):
    if index_type == 'Non-index':
      filename = './210123083_Data/Non-index_Stocks/' + stocks_name[i] + '.csv'
    else:
      filename = './210123083_Data/' + index_type[:3] + '_Stocks/' + stocks_name[i] + '.csv'
    df_stocks = pd.read_csv(filename)
    df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], errors='coerce')
    df_stocks.sort_values(by=['Date'], inplace=True)
    df_stocks.set_index('Date', inplace=True)

    daily_returns_stocks.append((df_stocks['Open'] - df_stocks['Price'])/df_stocks['Open'])
    

  beta_values = []
  for i in range(len(stocks_name)):
    df_combined = pd.concat([daily_returns_stocks[i], daily_returns], axis = 1, keys = [stocks_name[i], index_type])
    C = df_combined.cov()

    beta = C[index_type][stocks_name[i]]/C[index_type][index_type]
    beta_values.append(beta)

  return beta_values


def main():
  print("**********  Inference about stocks taken from BSE  **********")
  stocks_name_BSE = ['Asian_Paints', 'Axis_Bank', 'Bajaj_Finance', 'DRL', 'ICICI_Bank', 'Induslnd_Bank',
                'Nestle_India', 'Reliance_Industries', 'Sun_Pharma', 'Titan_Company']
  beta_BSE = compute_beta(stocks_name_BSE, './210123083_Data/BSESN.csv', 'BSE Index')
  mu_market_BSE, risk_market_BSE = find_market_portfolio('./210123083_Data/BSESN.csv')
  execute_model(stocks_name_BSE, 'BSE', mu_market_BSE, risk_market_BSE, beta_BSE)



  print("\n\n**********  Inference about stocks taken from NSE  **********")
  stocks_name_NSE = ['Apollo_Hospitals', 'Bajaj_Auto', 'Britannia_Industries', 'Hero_MotoCorp', 'Infosys',
            'ITC', 'LTIMindTree', 'Maruti_Suzuki', 'TCS', 'Wipro']
  beta_NSE = compute_beta(stocks_name_NSE, './210123083_Data/NSEI.csv', 'NSE Index')
  mu_market_NSE, risk_market_NSE = find_market_portfolio('./210123083_Data/NSEI.csv')
  execute_model(stocks_name_NSE, 'NSE', mu_market_NSE, risk_market_NSE, beta_NSE) 
    
    
    
  print("\n\n**********  Inference about stocks not taken from any index  with index taken from BSE values**********")
  stocks_name_non = ['Ambuja_Cements', 'Bosch', 'Dixon_Tech', 'Havells_India', 'InfoEdge_India',
            'InterGlobe_Aviation', 'Oracle', 'Procter_and_Gamble', 'Siemens_Ltd', 'Tube_Invest_India']
  beta_non_index_BSE = compute_beta(stocks_name_non, './210123083_Data/BSESN.csv', 'Non-index')
  execute_model(stocks_name_non, 'Non-index', mu_market_BSE, risk_market_BSE, beta_non_index_BSE) 


  print("\n\n**********  Inference about stocks not taken from any index  with index taken from NSE values**********")
  beta_non_index_NSE = compute_beta(stocks_name_non, './210123083_Data/NSEI.csv', 'Non-index')
  execute_model(stocks_name_non, 'Non-index', mu_market_NSE, risk_market_NSE, beta_non_index_NSE) 


if __name__=="__main__":
  main()