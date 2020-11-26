import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data

################################################### Dividend Yield ################################################

dy = pd.read_excel(r'DY.xlsx', parse_dates=True, index_col=0).dropna()
dy.astype(float)

def columns(df):
    df.index = df.index.str[-5:]
    return df
columns(dy)

dy.sort_values(by=['Dividend Yield'], ascending=False, inplace=True)
assets = dy.iloc[0:10]
assets.index

stocks = ['CGAS5.SA', 'ENAT3.SA', 'SEER3.SA', 'WIZS3.SA', 'ROMI3.SA', 'CYRE3.SA', 'DIRR3.SA', 'WHRL4.SA',
       'CSNA3.SA', 'GEPA4.SA', '^BVSP']
df = pd.DataFrame()
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-01')['Adj Close']

norm = pd.DataFrame()
for i in df:
    norm[i] = df[i].div(df[i].iloc[0]).mul(100)
norm.plot()
plt.legend(loc='lower left')
plt.show()
