import pandas as pd

# Read the two CSV files


filenames = [
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/AMBUJA/OPTSTK_AMBUJACEM_CE_28-Dec-2022_TO_29-Mar-2023.csv",
        "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/AMBUJA/OPTSTK_AMBUJACEM_PE_28-Dec-2022_TO_29-Mar-2023.csv"],
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/APOLLO/OPTSTK_APOLLOHOSP_CE_28-Dec-2022_TO_29-Mar-2023.csv",
        "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/APOLLO/OPTSTK_APOLLOHOSP_PE_28-Dec-2022_TO_29-Mar-2023.csv"],
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/MARUTI/OPTSTK_MARUTI_CE_28-Dec-2022_TO_29-Mar-2023.csv",
        "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/MARUTI/OPTSTK_MARUTI_PE_28-Dec-2022_TO_29-Mar-2023.csv"],
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/INDIGO/OPTSTK_INDIGO_CE_28-Dec-2022_TO_29-Mar-2023.csv",
        "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/INDIGO/OPTSTK_INDIGO_PE_28-Dec-2022_TO_29-Mar-2023.csv"],
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/TCS_options/OPTSTK_TCS_CE_28-Dec-2022_TO_29-Mar-2023.csv",
     "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/TCS_options/OPTSTK_TCS_PE_28-Dec-2022_TO_29-Mar-2023.csv"],
    ["/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/Nifty_options/OPTIDX_NIFTY_CE_29-Dec-2022_TO_29-Mar-2023.csv",
        "/Users/naveenkumar/Documents/OperationStrIX/MA374--3/Lab09/Nifty_options/OPTIDX_NIFTY_PE_29-Dec-2022_TO_29-Mar-2023.csv"]
]

output_filenames = ['ambujastockoptions.csv', 'apollostockoptions.csv',
                    'marutistockoptions.csv', 'indigostockoptions.csv', 'tcsstockoptions.csv', 'NIFTYoptiondata.csv']

for i in range(len(filenames)):
    df1 = pd.read_csv(filenames[i][0])
    df2 = pd.read_csv(filenames[i][1])

    # Merge the two DataFrames based on symbol, date, and expiry columns
    merged_df = pd.merge(
        df1, df2, on=['Symbol  ', 'Date  ', 'Expiry  ', 'Strike Price  ', 'Underlying Value  '])

    # Filter columns to include only the required ones
    merged_df = merged_df[['Symbol  ', 'Date  ', 'Expiry  ',
                           'Strike Price  ', 'Underlying Value  ', 'Close  _x', 'Close  _y']]

    # Rename columns for clarity
    merged_df.columns = ['symbol', 'date', 'expiry',
                         'strike price', 'underlying value', 'call price', 'put price']

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_filenames[i], index=False)

    print(f"{output_filenames[i]} done!")
