from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['boxplot.medianprops.color'] = 'C1'
mpl.rcParams['boxplot.meanprops.color'] = 'C0'


# Matplotlib

df = pd.read_csv('data/visual/ATLX Experiment Resutls - LX size UC3.csv', index_col=0)
df *= 100


_df = df[[c for c in df.columns if c.startswith('TEST')]]
x_labels = [0] + list(np.arange(200, 1200, 200)) + [1234]

X = _df.drop('TEST1200', axis=1).values



fig, ax = plt.subplots(figsize=(12,6))

bplot = ax.boxplot(X, whis=2, meanline=True, showmeans=True,
                   # patch_artist=True
                   )

ax.set_xticklabels(x_labels)
ax.set_xlabel('Lexicon size')
ax.set_ylabel('Acc. %')

plt.savefig(f"C:/Users/ebao/Desktop/ACL_images/test_acc_by_lx_size",bbox_inches='tight')





# Plotly
df = pd.read_csv('data/visual/ATLX Experiment Resutls - LX size UC3.csv', index_col=0)

df *= 100


_df = df[[c for c in df.columns if c.startswith('TEST')]]
x_labels = [0] + list(np.arange(200, 1200, 200)) + [1234]

X = _df.drop('TEST1200', axis=1).values


def make_trace(X, col, name):
    trace = go.Box(
        y = X[:,col],
        # x = name,
        name = 'n='+str(name),
        boxmean=True,
        showlegend=False
    )
    return trace

data = [make_trace(X, i, n) for i,n in zip(range(7),x_labels)]

layout = go.Layout(
    title='Acc. by lexicon size'
)

fig = go.Figure(data=data, layout=layout)
plot(data)







def make_trace(df, colname):
    return go.Box(
        y = df[colname].values,
        # x = ['Baseline', '200', '400', '600', '800', '1000', '1200', '1234'],
        name=name,
        boxmean=True,
        #boxpoints=False
        showlegend=False,
    )


data = [make_trace(df, c, ) for c in [_c for _c in df.columns if _c.startswith('TEST') and not _c.endswith('1200')]]
layout = go.Layout(
    # xaxis = dict(
    #     title = 'Acc. %',
    #     zeroline=False
    # )
)


fig = go.Figure(data=data, layout=layout)
plot(data)













data = [make_trace(df, c) for c in [_c for _c in df.columns if _c.startswith('DEV') and not _c.endswith('1200')]]

layout = go.Layout()


fig = go.Figure(data=data, layout=layout)
plot(data)
