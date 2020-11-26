import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pandas_datareader import data

df = pd.read_excel(r'Cotações.xlsx', parse_dates=True, index_col=0)

############################################################ Low Vol ##########################################################

def columns(df):
    df.columns = df.columns.str[-6:]
    return df
columns(df)
df.replace('-', np.nan, inplace=True)


df1 = pd.DataFrame()
ret = pd.DataFrame()

for i in df:
    df1[i] = df[i].dropna()
    ret[i] = df1[i].pct_change().dropna()

data = []
data = np.std(ret)*252**(1/2)*100
data = pd.DataFrame(data)
data.rename(columns={0:'Volatilidade'}, inplace=True)
data.dropna(inplace=True)
data.sort_values(by=['Volatilidade'], ascending=True, inplace=True)
assets = data.iloc[:10]
assets.index

stocks = ['TAEE11.SA', 'AGRO3.SA', 'CPFE3.SA', 'TIET11.SA', 'TRPL4.SA', 'CESP6.SA', 'ENGI11.SA', 'ALUP11.SA',
       'LEVE3.SA', 'EQTL3.SA']

### Portfolio returns 
df = pd.DataFrame()
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-01')['Adj Close']

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
plt.legend(['Portfolio - Low vol', 'Ibov'])
plt.show()