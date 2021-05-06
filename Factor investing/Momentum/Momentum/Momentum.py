import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt

df = pd.read_excel(r'Cotações.xlsx', parse_dates=True, index_col=0)

####################################################### Momentum ########################################################

def columns (df):
    df.columns = df.columns.str[-6:]
    return df
columns(df)
df.replace('-', np.nan, inplace=True)
df = df.dropna(axis=1, how='all')
df1 = df.dropna(axis=0, how='all')

### Para 2017

prices_2017 = df.loc['2017'].dropna(axis=1, how='all')
prices_2017 = prices_2017.loc['2017'].dropna(axis=0, how='all')
prices_2017
prices_2017_pct = {}

for stock in prices_2017:
    prices_2017_pct[stock] = prices_2017[stock].iloc[-1]/prices_2017[stock].iloc[0]*100

assests_2017 = pd.DataFrame([prices_2017_pct]).T
assests_2017.rename(columns={0:'PCT_2017'}, inplace=True)
assests_2017.sort_values(by=['PCT_2017'], ascending=False, inplace=True)
assests_2017.dropna(inplace=True)


### Para 2018

prices_2018 = df.loc['2018'].dropna(axis=1, how='all')
prices_2018 = prices_2018.loc['2018'].dropna(axis=0, how='all')

prices_2018_pct = {}

for stock in prices_2018:
    prices_2018_pct[stock] = prices_2018[stock].iloc[-1]/prices_2018[stock].iloc[0]*100

assests_2018 = pd.DataFrame([prices_2018_pct]).T
assests_2018.rename(columns={0:'PCT_2018'}, inplace=True)
assests_2018.sort_values(by=['PCT_2018'], ascending=False, inplace=True)
assests_2018.dropna(inplace=True)

### Para 2019

prices_2019 = df.loc['2019'].dropna(axis=1, how='all')
prices_2019 = prices_2019.loc['2019'].dropna(axis=0, how='all')
prices_2019_pct = {}

for stock in prices_2019:
    prices_2019_pct[stock] = prices_2019[stock].iloc[-1]/prices_2019[stock].iloc[0]*100

assests_2019 = pd.DataFrame([prices_2019_pct]).T
assests_2019.rename(columns={0:'PCT_2019'}, inplace=True)
assests_2019.sort_values(by=['PCT_2019'], ascending=False, inplace=True)
assests_2019.dropna(inplace=True)

momentum = pd.concat([assests_2017, assests_2018, assests_2019], axis=1)
momentum.sort_values(by=['PCT_2017', 'PCT_2018', 'PCT_2019'], ascending=False, inplace=True)
momentum.dropna(inplace=True)
momentum.index[19:22]

stocks = [
        '\nMGLU3', '\nUNIP6','\nPRIO3', '\nLCAM3', '\nGOLL4',
        '\nROMI3', '\nCRPG5', '\nTGMA3', '\nYDUQ3', '\nBPAC3'
       ]

### Teste de autocorrelação
autocorrel = {}
returns = {}

for i in stocks:
    returns[i] = df1[i].loc['2017':'2019'].pct_change().dropna()
    autocorrel[i] = returns[i].autocorr()

autocorrel
returns
bpac = df1['\nMGLU3'].pct_change().dropna()
bpac.autocorr()

#### Portfolio returns

stocks = [
        'MGLU3.SA', 'UNIP6.SA','PRIO3.SA', 'LCAM3.SA', 'GOLL4.SA',
        'ROMI3.SA', 'CRPG5.SA', 'TGMA3.SA', 'YDUQ3.SA', 'BPAC3.SA'
       ]

df = pd.DataFrame()
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-01', end = '2020-12-31')['Adj Close']

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
ibov = data.DataReader('^BVSP', data_source='yahoo', start='2020-01-01', end = '2020-12-31')
ibov.rename(columns = {'Adj Close':'IBOV'}, inplace=True)
ibov.drop(ibov.columns[[0,1,2,3,4]], axis=1, inplace=True)
ibov['Ibov'] = ibov['IBOV'].div(ibov['IBOV'].iloc[0]).mul(100)
ibov

plt.plot(norm['Portfolio'])
plt.plot(ibov['Ibov'])
plt.legend(['Portfolio - Momentum', 'Ibov'])
plt.show()

final = pd.concat([norm['Portfolio'], ibov['Ibov']], axis=1)
final.to_excel('Mome.xlsx')