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

data_lpa = pd.DataFrame.from_dict(data, orient='index')
data_lpa.sort_values(by=[0], ascending=False, inplace=True)
data_lpa