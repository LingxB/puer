import pandas as pd
from collections import Counter
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 80)


train = pd.read_csv('data/processed/SemEval14/SemEval14_train.csv')
test = pd.read_csv('data/processed/SemEval14/SemEval14_test.csv')
mpqa = pd.read_csv('data/processed/lexicon/MPQA.csv')
opener = pd.read_csv('data/processed/lexicon/opener.csv')
ol = pd.read_csv('data/processed/lexicon/opinion-lexicon-English.csv')
vader = pd.read_csv('data/processed/lexicon/vader.csv')


c_train = Counter()
for s in train.SENT.str.split():
    c_train.update(s)

c_test = Counter()
for s in test.SENT.str.split():
    c_test.update(s)

c_corpus = Counter()
for s in train.SENT.str.split():
    c_corpus.update(s)
for s in test.SENT.str.split():
    c_corpus.update(s)

train_words = set([w for w,_ in c_train.most_common()])
test_words = set([w for w,_ in c_test.most_common()])
corpus_words = set([w for w,_ in c_corpus.most_common()])

mpqa_words = set(mpqa.WORD.unique())
opener_words = set(opener.WORD.unique())
ol_words = set(ol.WORD.unique())
vader_words = set(vader.WORD.unique())


len(train_words) # 4435
len(test_words) # 2199
len(corpus_words) # 5175
len(opener_words) # 6885
len(mpqa_words) # 6886
len(ol_words) # 6787
len(vader_words) # 7503


def check_coverage(data: {}, lexicon: {} or []):
    if isinstance(lexicon, list):
        lexicon = lexicon[0].union(*lexicon)
    w_data = len(data)
    w_lexicon = len(lexicon)
    i = 0
    for w in data:
        if w in lexicon:
            i += 1
    print(f'Data: {w_data} tokens, lexicon: {w_lexicon} tokens, {i} words in data, coverage: {i/w_data:.2%}')


check_coverage(corpus_words, opener_words) # Data: 5175 tokens, lexicon: 6885 tokens, 908 words in data, coverage: 17.55%
check_coverage(corpus_words, mpqa_words) # Data: 5175 tokens, lexicon: 6886 tokens, 908 words in data, coverage: 17.55%
check_coverage(corpus_words, ol_words) # Data: 5175 tokens, lexicon: 6787 tokens, 732 words in data, coverage: 14.14%
check_coverage(corpus_words, vader_words) # Data: 5175 tokens, lexicon: 7503 tokens, 656 words in data, coverage: 12.68%

check_coverage(corpus_words, opener_words) # v4
# Data: 5175 tokens, lexicon: 6885 tokens, 908 words in data, coverage: 17.55%
check_coverage(corpus_words, [opener_words, mpqa_words]) # v5
#Data: 5175 tokens, lexicon: 6886 tokens, 908 words in data, coverage: 17.55%
check_coverage(corpus_words, [opener_words, mpqa_words, ol_words]) # v6
# Data: 5175 tokens, lexicon: 8259 tokens, 1073 words in data, coverage: 20.73%
check_coverage(corpus_words, [opener_words, mpqa_words, ol_words, vader_words]) # v2
# Data: 5175 tokens, lexicon: 13298 tokens, 1235 words in data, coverage: 23.86%