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
       'CSNA3.SA', 'GEPA4.SA']

df = pd.DataFrame()
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-01')['Adj Close']

### Portfolio returns
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
returns = df.pct_change().dropna()

df['Portfolio'] = (1+returns.dot(weights)).cumprod().dropna()

norm = pd.DataFrame()
for i in df:
    norm[i] = df[i].div(df[i].iloc[1]).mul(100)
norm

plt.style.use('ggplot')
norm.plot()
plt.legend(loc='lower left')
plt.show()

### Portfolio vs IBOV
ibov = data.DataReader('^BVSP', data_source='yahoo', start='2020-01-01')
ibov.rename(columns = {'Adj Close':'IBOV'}, inplace=True)
ibov.drop(ibov.columns[[0,1,2,3,4]], axis=1, inplace=True)
ibov['Ibov'] = ibov['IBOV'].div(ibov['IBOV'].iloc[0]).mul(100)
ibov

plt.plot(norm['Portfolio'])
plt.plot(ibov['Ibov'])
plt.legend(['Portfolio - DY', 'Ibov'])
plt.show()