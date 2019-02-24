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

w_in_corpus = lx_v2.WORD.apply(lambda w: w in corpus_words)

pd.value_counts(w_in_corpus)

effective_lexicon = lx_v2.loc[w_in_corpus].copy()

effective_lexicon = effective_lexicon.sample(frac=1, random_state=42).reset_index(drop=True)


effective_lexicon.loc[effective_lexicon.WORD == 'not'] # 1017
effective_lexicon.loc[effective_lexicon.WORD == "n't"] # 99


effective_lexicon.to_csv('data/processed/effective_lexicon/effective_lexicon.csv', index=False)
