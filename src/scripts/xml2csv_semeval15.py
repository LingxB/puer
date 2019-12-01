from xml.dom import minidom
import pandas as pd
import nltk
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def unpack_review(review, return_df=True):
    sents = review.getElementsByTagName('sentence')

    rev_dict = {
        'EA': [],
        'CLS': [],
        'SENT': []
    }

    for sent in sents:
        texts, ap_paris = unpack_sent(sent)
        rev_dict['SENT'] += texts
        rev_dict['EA'] += [a for a,p in ap_paris]
        rev_dict['CLS'] += [p for a, p in ap_paris]

    return pd.DataFrame(rev_dict) if return_df else rev_dict

def unpack_sent(sent):
    texts = [sent.getElementsByTagName('text')[0].firstChild.nodeValue]
    opinions = sent.getElementsByTagName('Opinion')
    aspect_polarity_pairs = []

    for opinion in opinions:
        aspect = opinion.attributes['category'].value
        polarity = opinion.attributes['polarity'].value
        aspect_polarity_pairs.append((aspect, polarity))

    if len(opinions) == 0:
        aspect_polarity_pairs.append((None, None))

    if len(texts) < len(aspect_polarity_pairs):
        texts = texts * ((len(aspect_polarity_pairs) - len(texts)) + 1)

    return texts, aspect_polarity_pairs


def xml2df(path):
    xml = minidom.parse(path)
    reviews = xml.getElementsByTagName('Review')
    rdfs = [unpack_review(r) for r in reviews]
    df = pd.concat(rdfs, ignore_index=True)
    return df

def process_sent(s):
    tokens = nltk.word_tokenize(s)
    tokens = [t.lower() for t in tokens]
    return ' '.join(tokens)

def process_df(df):
    _df = df.copy()
    # 1. Drop None
    _df = _df.loc[~_df.isna().any(axis=1)]
    # 2. replace CLS with value
    _df['CLS'] = _df.CLS.replace(['positive', 'neutral', 'negative'], [1, 0, -1])
    # 3. process sent
    _df['SENT'] = _df.SENT.apply(process_sent)
    # 4. Extract aspect (attribute), simplify aspect term
    _df['ASP'] = _df.EA.apply(lambda s: s.split('#')[-1])
    _df['ASP'] = _df.ASP.replace(['OPERATION_PERFORMANCE', 'DESIGN_FEATURES', 'STYLE_OPTIONS'],
                                 ['PERFORMANCE', 'DESIGN', 'OPTIONS'])
    _df['ASP'] = _df.ASP.str.lower()
    return _df






# LAPTOP TRAIN
# ------------
train_df = xml2df('data/raw/semeval15/laptop/ABSA15_LaptopsTrain/ABSA-15_Laptops_Train_Data.xml')

train = process_df(train_df)

train.to_csv('data/processed/SemEval15_laptop/train.csv', index=False)


# LAPTOP TEST
test_df = xml2df('data/raw/semeval15/laptop/ABSA15_Laptops_Test.xml')

test = process_df(test_df)

test.to_csv('data/processed/SemEval15_laptop/test.csv', index=False)






# RESTRUANT TRAIN
# ---------------
train_df = xml2df('data/raw/semeval15/restruant/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml')

train = process_df(train_df)

train.to_csv('data/processed/SemEval15_rest/train.csv', index=False)

# RESTRUANT TEST
test_df = xml2df('data/raw/semeval15/restruant/ABSA15_Restaurants_Test.xml')

test = process_df(test_df)

test.to_csv('data/processed/SemEval15_rest/test.csv', index=False)

