import pandas as pd
from collections import Counter

train = pd.read_csv('data/processed/SemEval14/SemEval14_train.csv')
test = pd.read_csv('data/processed/SemEval14/SemEval14_test.csv')

train.SENTS.str.split()