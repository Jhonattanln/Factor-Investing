import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

########################################################## Size ###################################################################

size = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Size\Size.xlsx', 
                     parse_dates=True, index_col=0).dropna()
def columns(df):
    df.columns = df.columns.str[-6:]
    return df

columns(size)
size = size.T
size.replace('-', np.nan, inplace=True)

size.columns = size.columns.astype(str)
size.rename(columns={'2019-12-31': 'Market Cap'}, inplace=True)
size.sort_values('Market Cap', inplace=True)

assets = size.iloc[4:14]
assets.index

stocks = ['ATOM3.SA', 'IGBR3.SA', 'FHER3.SA', 'PDGR3.SA', 'SLED4.SA', 'BTTL3.SA',
       'VIVR3.SA', 'RCSL4.SA', 'ETER3.SA', 'RSID3.SA']

df = pd.DataFrame()
from pandas_datareader import data
for i in stocks:
    df[i] = data.DataReader(i, data_source='yahoo', start='2020-01-02', end = '2020-12-31')['Adj Close']

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
plt.legend(['Portfolio - Size', 'Ibov'])
plt.show()

final = pd.concat([norm['Portfolio'], ibov['Ibov']], axis=1)
final.to_excel('teste.xlsx', sheet_name = 'Size')
writer = pd.ExcelWriter('final.xlsx')
final.to_excel(writer)
writer.save()