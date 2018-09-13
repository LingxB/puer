import pandas as pd
import xml.etree.ElementTree


pd.options.display.max_colwidth = 80




e = xml.etree.ElementTree.parse('data/raw/opener_VUSentimentLexicon_en_general.xml')


entries = e.findall('Lexicon')[0].findall('LexicalEntry')


# # word
# entries[0].find('Lemma').attrib['writtenForm']

# # polarity
# entries[0].find('Sense').find('Sentiment').attrib['polarity']


words = []
polarities = []


for ent in entries:
    w = ent.find('Lemma').attrib['writtenForm']
    p = ent.find('Sense').find('Sentiment').attrib['polarity']
    words.append(w)
    polarities.append(p)


df = pd.DataFrame({'WORD': words, 'POL': polarities})

converter = {'positive': 1,
             'negative': -1,
             'neutral': 0,
             'both': 0,
             }

df['OPENER'] = df['POL'].apply(lambda x: converter[x])

df = df[['WORD', 'OPENER', 'POL']]

df.to_csv('data/processed/lexicon/opener.csv', index=False)