import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

########################################################## Size ###################################################################

size = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Size\Size.xlsx', 
                     parse_dates=True, index_col=0).dropna()
def columns(df):
    df.columns = df.columns.str[-6:]
    return df

columns(size)
size = size.T

size.rename(columns={2019-12-31: 'Market Cap'}, inplace=True)
size