import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

######################################################### D/E #############################################################
de = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Quality\Quality\DE.xlsx',
                   parse_dates=True, index_col=0)

def columns(df):
    df.columns = df.columns.str[-6:]
    return df

columns(de)
de.replace('-', np.nan, inplace=True)
de = de.T
de.columns = de.columns.astype(str)
de.rename(columns={'Data': 'Ação', '2019-12-31':'DE'}, inplace=True)
data_de = de[de['DE'] > 0]
data_de.sort_values('DE')

######################################################### ROE ########################################################
roe = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Quality\Quality\ROE.xlsx', 
                    parse_dates=True, index_col=0)

columns(roe)
roe.replace('-',np.nan, inplace=True)
roe = roe.T
roe.columns = roe.columns.astype(str)
roe.rename(columns={'2019-12-31':'ROE'}, inplace=True)
data_roe = roe[roe['ROE']>0]
data_roe.sort_values(by=['ROE'], ascending=False, inplace=True)
data_roe

############# Concatenando dados

data=pd.concat([data_de, data_roe], axis=1).dropna()

############# Ranking

de_value = data.DE.sum()
roe_value = data.ROE.sum()

values={}

for i in data.DE:
    values['DE'] = data.DE /de_value

values = pd.DataFrame(values)

for i in data.ROE:
    values['ROE'] = data['ROE'] / roe_value

values['Ranking'] = values.DE.div(values.ROE)
values.sort_values(by=['Ranking'], inplace=True)
assets = values.iloc[:20]
assets.index

stocks = ['ITSA4.SA', 'CEBR5.SA', 'LREN3.SA', 'TRPL4.SA', 'PARD3.SA', 'HAPV3.SA', 'STBP3.SA', 'TGMA3.SA',
       'CEAB3.SA', 'UNIP3.SA',]

### Potfolio

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
plt.legend(['Portfolio - Quality', 'Ibov'])
plt.show()