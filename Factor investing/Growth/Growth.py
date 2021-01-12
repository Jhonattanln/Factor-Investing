import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

lpa = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Growth\LPA.xlsx',
                    parse_dates=True, index_col=0)

###################################################### LPA ########################################################

def columns(df):
    df.columns = df.columns.str[-6:]
    return df
columns(lpa)

lpa.replace('-', np.nan, inplace=True)

lpa = lpa.pct_change()

data_lpa = {}

for i in lpa:
    data_lpa[i] = lpa[i].mean()

data_lpa = pd.DataFrame.from_dict(data_lpa, orient='index')
data_lpa.sort_values(by=[0], ascending=False, inplace=True)
data_lpa.rename(columns={0:'LPA'}, inplace=True)

##################################################### Sales #######################################################

sales = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Projetos\Factor Investing\Factor-Investing\Factor investing\Growth\Sales.xlsx',
                    parse_dates=True, index_col=0)

columns(sales)

sales.replace('-', np.nan, inplace=True)

data_sales = {}

for i in sales:
    data_sales[i] = sales[i].mean()


data_sales = pd.DataFrame.from_dict(data_sales, orient='index')
data_sales.sort_values(by=[0], ascending=False, inplace=True)
data_sales.rename(columns={0:'Price Sales Ratio'}, inplace=True)



############### Concatendo dados
data = pd.concat([data_lpa, data_sales], axis=1).dropna()
print(data)

lpa_value = data.LPA.sum()
sales_value = data['Price Sales Ratio'].sum()

values = {}

for i in data.LPA:
    values['LPA'] = data.LPA / lpa_value

values = pd.DataFrame(values)

for i in data['Price Sales Ratio']:
    values['Sales'] = data['Price Sales Ratio'] / sales_value

values['Ranking'] = values.LPA.div(values.Sales)
values.sort_values(by=['Ranking'], inplace=True)
assets = values.iloc[:20]
assets.index

stocks = ['POSI3.SA', 'JBSS3.SA', 'CSNA3.SA', 'RDNI3.SA', 'EMBR3.SA', 'USIM5.SA',
       'CEAB3.SA', 'ENAT3.SA', 'PRIO3.SA', 'LEVE3.SA']

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