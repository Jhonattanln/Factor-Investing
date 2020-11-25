import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################################################### Dividend Yield ################################################

dy = pd.read_excel(r'DY.xlsx', parse_dates=True, index_col=0).dropna()
dy.astype(float)

def columns(df):
    df.index = df.index.str[-5:]
    return df
columns(dy)

dy.sort_values(by=['Dividend Yield'], ascending=False, inplace=True)
assets = dy.iloc[0:10]
assets
