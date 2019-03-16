from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import pandas as pd
import numpy as np


df = pd.read_csv('data/visual/ATLX Experiment Resutls - LX size UC3.csv', index_col=0)

df *= 100




def make_trace(df, colname):
    return go.Box(
        y = df[colname].values,
        # x = ['Baseline', '200', '400', '600', '800', '1000', '1200', '1234'],
        name=colname,
        boxmean=True,
        #boxpoints=False
        showlegend=False,
    )


data = [make_trace(df, c) for c in [_c for _c in df.columns if _c.startswith('DEV') and not _c.endswith('1200')]]

layout = go.Layout()


fig = go.Figure(data=data, layout=layout)
plot(data)





data = [make_trace(df, c) for c in [_c for _c in df.columns if _c.startswith('TEST') and not _c.endswith('1200')]]
layout = go.Layout()


fig = go.Figure(data=data, layout=layout)
plot(data)
