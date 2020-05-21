from pathlib import Path
import sys
parent = str(Path(__file__).parent.parent.parent.absolute())
sys.path.append(parent)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from src.visualization.plots import str2array, lx_get



def loc_sent(df, idx):
    return df.iloc[idx].to_frame().transpose()

@st.cache
def load_all():
    baseline = pd.read_csv('data/score/dalx/baseline.csv')
    general = pd.read_csv('data/score/dalx/general.csv')
    gold = pd.read_csv('data/score/dalx/gold_v2.csv')
    dalx_bin = pd.read_csv('data/score/dalx/dalx_binary.csv')
    dalx_tre = pd.read_csv('data/score/dalx/dalx_3way.csv')

    lx_general = pd.read_csv('data/processed/lexicon_v2/lexicon_table_v2.csv', index_col='WORD')
    lx_gold = pd.read_csv('data/processed/dalx/lexicon_table_dalx_15_gold_v2.csv', index_col='WORD')
    lx_dalx_bin = pd.read_csv('data/processed/dalx/lexicon_table_dalx_07_thres0.7_C10.csv', index_col='WORD')
    lx_dalx_tre = pd.read_csv('data/processed/dalx/lexicon_table_dalx_17_thres0.7_C10.csv', index_col='WORD')

    s15laptop = pd.read_csv('data/processed/SemEval15_laptop/test.csv')

    return (baseline, general, gold, dalx_bin, dalx_tre), \
           (lx_general, lx_gold, lx_dalx_bin, lx_dalx_tre), \
           s15laptop


def plot_sent(s: list, alpha: list, lx: list = None, figsize=None):
    x = s
    if lx is not None:
        z = np.stack([alpha, lx]).round(2)
        y = ['Alpha', 'LX']
    else:
        z = np.round(alpha, 2).reshape(1, -1)
        y = ['Alpha']

    fig, ax = plt.subplots(figsize=figsize)

    if len(z) > 1:
        for i in list(range(len(z)))[::-1]:
            _z = z.copy()
            _z[i, :] = np.nan
            ax.imshow(_z, cmap=cm.viridis if i != 0 else cm.RdGy)
    else:
        ax.imshow(z, cmap=cm.viridis)

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.xaxis.set_ticks_position('top')

    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y)

    d_text = np.zeros(z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, z[i, j], ha='center', va='center', color='k' if d_text[i, j] else 'w')

    st.pyplot()


models, lexicons, s15laptop = load_all()
baseline, general, gold, dalx_bin, dalx_tre = models
model_names = ['baseline', 'general', 'gold', 'dalx_bin', 'dalx_tre']
# lx_general, lx_gold, lx_dalx_bin, lx_dalx_tre = lexicons
lexicons = [None] + list(lexicons)


idx = st.number_input('Example index', value=0)
sent = loc_sent(s15laptop, idx)
s = sent.SENT.iloc[0]
st.write(sent)
st.write(s)


for model, model_name, lexicon in zip(models, model_names, lexicons):
    st.write(f"## {model_name}")
    pred = model.iloc[idx].PRED
    st.write(f"Label: {sent.CLS.iloc[0]}; Prediction: {pred}")
    sl = s.split()
    alpha = str2array(model.iloc[idx].ALPHA)[:len(sl)]
    if lexicon is not None:
        lx = np.array([lx_get(lexicon, w).mean() for w in sl])
    else:
        lx = None
    plot_sent(sl, alpha, lx, figsize=(12,2))
#
# model = baseline
#
# sl = s.split()
# pred = model.iloc[idx].PRED
# alpha = str2array(model.iloc[idx].ALPHA)[:len(sl)]
# lx = np.array([lx_get(lx_general, w).mean() for w in sl])
# plot_sent(sl, alpha, lx)
#
#
#
#
#
#
#
#
#
# for model, model_name in zip(models, model_names):
#     st.write(f"## {model_name}")
#     pred, prob, alpha = score_sent(model, sent)
#     st.write(f"Label: {sent.CLS.iloc[0]}; Prediction: {pred}")
#     lx = model.dm.lm.loc_word_pol(s.split()).mean(axis=1)
#     plot_sent(s.split(), alpha, lx, figsize=(12, 2))
#
#


# pred, prob, alpha = score_sent(baseline, sent)
# lx = baseline.dm.lm.loc_word_pol(s.split()).mean(axis=1)
#
# plot_sent(s.split(), alpha, lx, figsize=(12,8))




#
# s = sent.SENT.iloc[0].split()
# s_pol = baseline.dm.lm.loc_word_pol(s).mean(axis=1)
# z = np.stack([alpha, s_pol]).round(2)
# x = s
# y = ['Alpha', 'LX']
#
# fig, ax = plt.subplots(figsize=(12,8))
#
# _z = z.copy()
# _z[1,:] = np.nan
# ax.imshow(_z, cmap=cm.viridis)
#
# _z = z.copy()
# _z[0,:] = np.nan
# ax.imshow(_z, cmap=cm.RdGy)
#
#
# ax.set_xticks(np.arange(len(x)))
# ax.set_xticklabels(x)
# ax.xaxis.set_ticks_position('top')
#
# ax.set_yticks(np.arange(len(y)))
# ax.set_yticklabels(y)
#
# d_text = np.zeros(z.shape)
# for i in range(z.shape[0]):
#     for j in range(z.shape[1]):
#         ax.text(j, i, z[i,j], ha='center', va='center', color='k' if d_text[i,j] else 'w')


# sent = loc_sent(s15laptop, 0)
#
# pred, prob, alpha = score_sent(baseline, sent)