import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


################################################### Preço Lucro #########################################################

pl = pd.read_excel(r'PL.xlsx', parse_dates=True, index_col=0)

def columns (df):
    df.columns = df.columns.str[-6:]
    return df
columns(pl)
pl.replace('-', np.nan, inplace=True)

pl = pl.T

data_PL = pl[pl['2019-12-31']>0].dropna()
data_PL.sort_values(by=['2019-12-31'], ascending=True, inplace=True)
data_PL.columns = ['PL']

################################################### P/ VPA ##############################################################

vpa = pd.read_excel(r'PVPA.xlsx', parse_dates=True, index_col=0)

columns(vpa)
vpa.replace('-', np.nan, inplace = True)

vpa = vpa.T

data_VPA = vpa[vpa['2019-12-31']>0].dropna()
data_VPA.sort_values(by=['2019-12-31'], ascending=True, inplace=True)
data_VPA.columns = [ 'P/VPA']
data_VPA

##### Concatenando dados

data = pd.concat([data_PL, data_VPA], axis=1)
data.sort_values(['PL', 'P/VPA']).dropna()
data.index
data.PL.loc['\nFESA4']

### Ranking das açoes 

pl_value = data.PL.sum()
pva_value = data['P/VPA'].sum()
values = {}

for i in data.PL:
    values['PL'] = data.PL / pl_value

values = pd.DataFrame(values)

for i in data['P/VPA']:
    values['P/VPA'] = data['P/VPA'] / pva_value

values = values[['PL', 'P/VPA']].dropna()
values['Ranking'] = values.sum(axis=1)
values.sort_values('Ranking', inplace=True)
values
### Escolhas das ações

assets = values.iloc[:10]
assets.index

stocks = ['ELET3.SA', 'BMEB4.SA', 'CGRA4.SA', 'LIGT3.SA', 'FESA4.SA', 'BRSR6.SA',
       'EUCA4.SA', 'ABCB4.SA', 'SGPS3.SA', 'CMIG4.SA']

### Potfolio

df = pd.DataFrame()
from pandas_datareader import data
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-01', end = '2020-12-31')['Adj Close']

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

ibov = data.DataReader('^BVSP', data_source='yahoo', start='2020-01-01', end = '2020-12-31')
ibov.rename(columns = {'Adj Close':'IBOV'}, inplace=True)
ibov.drop(ibov.columns[[0,1,2,3,4]], axis=1, inplace=True)
ibov['Ibov'] = ibov['IBOV'].div(ibov['IBOV'].iloc[0]).mul(100)
ibov

plt.plot(norm['Portfolio'])
plt.plot(ibov['Ibov'])
plt.legend(['Portfolio - Value', 'Ibov'])
plt.show()

final = pd.concat([norm['Portfolio'], ibov['Ibov']], axis=1)
final.to_excel('Value.xlsx')