import pandas as pd

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
  print("**********  Beta for securities in BSE  **********")
  stocks_name_BSE = ['Asian_Paints', 'Axis_Bank', 'Bajaj_Finance', 'DRL', 'ICICI_Bank', 'Induslnd_Bank',
                'Nestle_India', 'Reliance_Industries', 'Sun_Pharma', 'Titan_Company']
  beta_BSE = compute_beta(stocks_name_BSE, './210123083_Data/BSESN.csv', 'BSE Index')

  for i in range(len(beta_BSE)):
    print(f"{stocks_name_BSE[i] : <20}= {beta_BSE[i]}")



  print("\n\n**********  Beta for securities in NSE  **********")
  stocks_name_NSE = ['Apollo_Hospitals', 'Bajaj_Auto', 'Britannia_Industries', 'Hero_MotoCorp', 'Infosys',
            'ITC', 'LTIMindTree', 'Maruti_Suzuki', 'TCS', 'Wipro']
  beta_NSE = compute_beta(stocks_name_NSE, './210123083_Data/NSEI.csv', 'NSE Index')
  
  for i in range(len(beta_NSE)):
    print(f"{stocks_name_NSE[i] : <20}= {beta_NSE[i]}")
    
    

  print("\n\n**********  Beta for securities in non-index using BSE Index  **********")
  stocks_name_non = ['Ambuja_Cements', 'Bosch', 'Dixon_Tech', 'Havells_India', 'InfoEdge_India',
            'InterGlobe_Aviation', 'Oracle', 'Procter_and_Gamble', 'Siemens_Ltd', 'Tube_Invest_India']
  beta_non_BSE = compute_beta(stocks_name_non, './210123083_Data/BSESN.csv', 'Non-index')
  
  for i in range(len(beta_non_BSE)):
    print(f"{stocks_name_non[i] : <20}= {beta_non_BSE[i]}")
    
  

  print("\n\n**********  Beta for securities in non-index using NSE Index  **********")
  beta_non_NSE = compute_beta(stocks_name_non, './210123083_Data/NSEI.csv', 'Non-index')
  
  for i in range(len(beta_non_NSE)):
    print(f"{stocks_name_non[i] : <20}= {beta_non_NSE[i]}")


if __name__=="__main__":
  main()