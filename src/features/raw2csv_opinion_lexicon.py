import pandas as pd


positive_words = 'data/raw/opinion-lexicon-English/positive-words.txt'
negative_words = 'data/raw/opinion-lexicon-English/negative-words.txt'
output_dir = 'data/processed/lexicon'



def read_opinion_txt(file, start_row, encoding='utf-8'):
    rows = []
    with open(file, 'r', encoding=encoding) as f:
        for idx,line in enumerate(f):
            try:
                if idx + 1 >= start_row:
                    rows.append(line.strip())
            except UnicodeDecodeError:
                print(idx+1)
    return rows

w_pos = read_opinion_txt(positive_words, 36)
w_neg = read_opinion_txt(negative_words, 36, encoding='ISO-8859-1')


df_pos = pd.DataFrame({'WORD': w_pos, 'OL': [1] * len(w_pos)})
df_neg = pd.DataFrame({'WORD': w_neg, 'OL': [-1] * len(w_neg)})

ol_df = pd.concat([df_pos, df_neg], ignore_index=True)

ol_df.to_csv(output_dir+'/opinion-lexicon-English.csv')
