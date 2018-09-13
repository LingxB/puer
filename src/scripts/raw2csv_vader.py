import pandas as pd
pd.options.display.max_colwidth = 80


vader_file = 'data/raw/vader_lexicon.txt'



word = []
pol = []
std = []

with open(vader_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        w = line.strip().split('\t')[0]
        p = float(line.strip().split('\t')[1])
        s = float(line.strip().split('\t')[2])
        word.append(w)
        pol.append(p)
        std.append(s)


vader = pd.DataFrame({'WORD': word,
                      'VADER': pol,
                      'STD': std
                      })

# Standarlize polarity scores, vader was labeled on a scale from "[–4] Extremely Negative" to "[4] Extremely Positive"
# https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions

vader.VADER /= 4

vader.to_csv('data/processed/lexicon/vader.csv', index=False)
