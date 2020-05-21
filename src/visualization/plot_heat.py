import pandas as pd
import numpy as np
from src.visualization.plots import str2array, lx_get
import matplotlib.pyplot as plt
import matplotlib.cm as cm



base = pd.read_csv('data/score/SemEval14_test_baseline_v1.csv')
atlx = pd.read_csv('data/score/SemEval14_test_exp_3_lx2_uc3.csv')
lexicon = pd.read_csv('data/processed/lexicon_v2/lexicon_table_v2.csv', index_col='WORD')
ent = pd.read_csv('data/score/SemEval14_test_exp_17.csv')
ent_pos = pd.read_csv('data/score/SemEval14_test_exp_12z_base_ent+0.025.csv')

exp = 516

b = base.loc[exp]
a = atlx.loc[exp]
e = ent.loc[exp]
ep = ent_pos.loc[exp]

sent = b.SENT.split()
ba = str2array(b.ALPHA)[:len(sent)]
lx = np.array([lx_get(lexicon, w).mean() for w in sent])
aa = str2array(a.ALPHA)[:len(sent)]
ea = str2array(e.ALPHA)[:len(sent)]
epa = str2array(ep.ALPHA)[:len(sent)]

z = np.stack([ba, ea, epa]).round(2)
x = sent
y = ['base: Pos', 'base_ent-: Pos', 'base_ent+: Neg']


fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(z, cmap=cm.viridis)

#
# _z = z.copy()
# _z[-1,:] = np.nan
# _z[-3,:] = np.nan
# ax.imshow(_z, cmap=cm.viridis)
#
# _z = z.copy()
# _z[:2,:] = np.nan
# _z[3:,:] = np.nan
# ax.imshow(_z, cmap=cm.viridis)
#
#
# _z = z.copy()
# _z[:-1,:] = np.nan
# ax.imshow(_z, cmap=cm.RdGy)




ax.set_xticks(np.arange(len(x)))
ax.set_xticklabels(x)
ax.xaxis.set_ticks_position('top')

ax.set_yticks(np.arange(len(y)))
ax.set_yticklabels(y)

d_text = np.zeros(z.shape)
d_text[-1,4] = 1

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, z[i,j], ha='center', va='center', color='k' if d_text[i,j] else 'w')

ax.set_xlabel('LABEL: Pos / ASPECT: food')



plt.savefig(f"C:/Users/ebao/Desktop/ACL_images/base_ent+-_atlx_e{exp}",bbox_inches='tight')






# ---
base = pd.read_csv('data/score/SemEval14_test_baseline_v1.csv')
atlx = pd.read_csv('data/score/SemEval14_test_exp_3_lx2_uc3.csv')
lexicon = pd.read_csv('data/processed/lexicon_v2/lexicon_table_v2.csv', index_col='WORD')
std = pd.read_csv('data/score/SemEval14_test_exp_15.csv')
ent = pd.read_csv('data/score/SemEval14_test_exp_17.csv')
ent_pos = pd.read_csv('data/score/SemEval14_test_exp_12z_base_ent+0.025.csv')

exp = 941

b = base.loc[exp]
a = atlx.loc[exp]
s = std.loc[exp]
e = ent.loc[exp]
ep = ent_pos.loc[exp]

sent = b.SENT.split()
ba = str2array(b.ALPHA)[:len(sent)]
lx = np.array([lx_get(lexicon, w).mean() for w in sent])
aa = str2array(a.ALPHA)[:len(sent)]
sa = str2array(s.ALPHA)[:len(sent)]
ea = str2array(e.ALPHA)[:len(sent)]
epa = str2array(ep.ALPHA)[:len(sent)]

z = np.stack([ba, sa, ea, epa, aa, lx]).round(2)
x = sent
y = ['base: Pos', 'base_std: Neg', 'base_ent-: Neg', 'base_ent+: Neg', 'ATLX: Neg', 'lexicon']


fig, ax = plt.subplots(figsize=(12,8))

_z = z.copy()
_z[-1,:] = np.nan
_z[-3,:] = np.nan
ax.imshow(_z, cmap=cm.viridis)

_z = z.copy()
_z[:3,:] = np.nan
_z[4:,:] = np.nan
ax.imshow(_z, cmap=cm.viridis)


_z = z.copy()
_z[:-1,:] = np.nan
ax.imshow(_z, cmap=cm.RdGy)




ax.set_xticks(np.arange(len(x)))
ax.set_xticklabels(x)
ax.xaxis.set_ticks_position('top')

ax.set_yticks(np.arange(len(y)))
ax.set_yticklabels(y)


d_text = np.zeros(z.shape)
d_text[0,-2:] = 1
d_text[1,-2] = 1
d_text[-1,:] = 1
d_text[3,7] = 1
for idx in [9, 17, 18]:
    d_text[-1,idx] = 0

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, z[i,j], ha='center', va='center', color='k' if d_text[i,j] else 'w')

ax.set_xlabel('LABEL: Neg / ASPECT: service')



plt.savefig(f"C:/Users/ebao/Desktop/ACL_images/base_bstd_bent+-_atlx_e{exp}",bbox_inches='tight')














# ----
base = pd.read_csv('data/score/SemEval14_test_baseline_v1.csv')
atlx = pd.read_csv('data/score/SemEval14_test_exp_3_lx2_uc3.csv')
lexicon = pd.read_csv('data/processed/lexicon_v2/lexicon_table_v2.csv', index_col='WORD')



plt.rcParams.update({'font.size': 17})


exp = 228 #475 #228

b = base.loc[exp]
a = atlx.loc[exp]


sent = b.SENT.split()
ab = str2array(b.ALPHA)[:len(sent)]
lx = np.array([lx_get(lexicon, w).mean() for w in sent])
aa = str2array(a.ALPHA)[:len(sent)]



# Matplotlib
z = np.stack([ab, aa, lx]).round(2)
x = sent
y = ['Base: Pos', 'ATLX: Neg', 'Lexicon']



fig, ax = plt.subplots(figsize=(12,8))

_z = z.copy()
_z[-1,:] = np.nan
ax.imshow(_z, cmap=cm.viridis)

_z = z.copy()
_z[:-1,:] = np.nan
ax.imshow(_z, cmap=cm.RdGy)

ax.set_xticks(np.arange(len(x)))
ax.set_xticklabels(x)
ax.xaxis.set_ticks_position('top')

ax.set_yticks(np.arange(len(y)))
ax.set_yticklabels(y)


d_text = np.zeros(z.shape)

# exp 228
d_text[1,-2] = 1
for idx in [0, 1, 4, 6]:
    d_text[-1,idx] = 1

# exp 475
# d_text[:2,3] = 1

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, z[i,j], ha='center', va='center', color='k' if d_text[i,j] else 'w')

ax.set_xlabel('LABEL: Neg / ASPECT: ambiance')

plt.savefig(f"C:/Users/ebao/Desktop/ACL_images/ATLX_vs_base_e{exp}",bbox_inches='tight')


























