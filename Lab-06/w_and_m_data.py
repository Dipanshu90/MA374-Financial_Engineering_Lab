import pandas as pd

df = pd.read_csv('./BSE/dbsedata1.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)
df = df.resample('W-Tue').first()
df.drop(df.tail(1).index,inplace=True)
df.to_csv('./BSE/wbsedata1.csv')

df = pd.read_csv('./BSE/dbsedata1.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)
df = df.resample('MS').first()
df.to_csv('./BSE/mbsedata1.csv')

df = pd.read_csv('./NSE/dnsedata1.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)
df = df.resample('W-Tue').first()
df.drop(df.tail(1).index,inplace=True)
df.to_csv('./NSE/wnsedata1.csv')

df = pd.read_csv('./NSE/dnsedata1.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)
df = df.resample('MS').first()
df.to_csv('./NSE/mnsedata1.csv')
