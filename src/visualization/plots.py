import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def str2array(string):
    return np.array([float(x.strip()) for x in string.replace('[', '').replace(']', '').split(',')])

def lx_get(df, key, default=0):
    try:
        return df.loc[key]
    except KeyError:
        return np.array(default)

def plot_sent(tokens: list, alpha: str or list or np.ndarray, lexicon: pd.DataFrame):
    lx = np.array([lx_get(lexicon, w).mean() for w in tokens])
    if isinstance(alpha, str):
        attention = str2array(alpha)[:len(tokens)]
    else:
        attention = alpha
    assert len(tokens) == len(attention)
    data = pd.DataFrame({'Lexicon': lx, 'Attention': attention})

    fig, ax = plt.subplots()
    ax.matshow(data.mask(((data == data) | data.isnull()) & (data.columns != "Attention")).transpose(), cmap=cm.viridis)
    ax.matshow(data.mask(((data == data) | data.isnull()) & (data.columns != "Lexicon")).transpose(), cmap=cm.cividis)
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['Lexicon', 'Attention'])

    for i in range(len(tokens)):
        for j in range(2):
            ax.text(i, j, round(data.iloc[i, j], 2), ha='center', va='center', color='w')


#
# df = pd.read_csv('data/score/SemEval14_test_baseline_v1.csv')
# lexicon = pd.read_csv('data/processed/lexicon/lexicon_table.csv', index_col='WORD')
# s1 = df.iloc[0]
# tokens = s1.SENT.split()
#
# plot_sent(s1.SENT.split(), s1.ALPHA, lexicon)




