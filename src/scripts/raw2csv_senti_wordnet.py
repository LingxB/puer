import pandas as pd

pd.options.display.max_colwidth = 80


sentiwordnet = 'data/raw/sentiwordnet/SentiWordNet_3.0.0_20130122.txt'


df = pd.read_csv(sentiwordnet, sep='\t', skiprows=26)

df['SWN'] = df.PosScore - df.NegScore

tmp = df[['SynsetTerms', 'SWN', 'PosScore', 'NegScore', '# POS']].copy()



swn = pd.DataFrame(tmp.SynsetTerms.str.split(' ').tolist(), index=tmp.SWN).stack().reset_index()[[0, 'SWN']]
swn.columns = ['WORD', 'SWN']

swn['WORD'] = swn.WORD.str.replace(r'#\d', '')
swn.dropna(inplace=True)
swn = swn.groupby('WORD').mean().reset_index()

swn.to_csv('data/processed/lexicon/SentiWordNet.csv', index=False)