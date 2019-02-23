"""
Script to filter lexicon_v2 for SemEval14 data and produce an effective lexicon for lexicon size experiment
"""

import pandas as pd
from collections import Counter
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 80)


train = pd.read_csv('data/processed/SemEval14/SemEval14_train.csv')
test = pd.read_csv('data/processed/SemEval14/SemEval14_test.csv')
lx_v2 = pd.read_csv('data/processed/lexicon_v2/lexicon_table_v2.csv')


c_corpus = Counter()
for s in train.SENT.str.split():
    c_corpus.update(s)
for s in test.SENT.str.split():
    c_corpus.update(s)

corpus_words = set([w for w,_ in c_corpus.most_common()])

