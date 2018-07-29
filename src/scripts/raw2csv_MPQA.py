import pandas as pd


in_file = 'data/raw\subjectivity_clues_hltemnlp05\subjclueslen1-HLTEMNLP05.tff'
out_dir = 'data/processed/lexicon'

def read_MPOA(file):
    content = []
    with open(file, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            row = line.strip().split()
            d = {x.split('=')[0]:x.split('=')[1] for x in row if x != 'm'} # Some rows have 'm' without '='
            t = d.get('type')
            l = d.get('len')
            w = d.get('word1')
            pos = d.get('pos1')
            s = d.get('stemmed1')
            p = d.get('polarity')
            pp = d.get('priorpolarity')
            mp = d.get('mpqapolarity')
            content.append([t, l, w, pos, s, p, pp, mp])
    return content

content = read_MPOA(in_file)


mpqa_df = pd.DataFrame(content, columns=['TYPE','LEN','WORD','POS','STEM','P','PP','MP'])

converter = {'positive': 1,
             'negative': -1,
             'neutral': 0,
             'both': 0,
             'weakneg': -0.5
             }

mpqa_df['MPQA'] = mpqa_df['PP'].apply(lambda x: converter[x])

mpqa_df = mpqa_df[['WORD','MPQA','TYPE','LEN','POS','STEM','P','PP','MP']]

mpqa_df.to_csv(out_dir + '/MPQA.csv', index=False)