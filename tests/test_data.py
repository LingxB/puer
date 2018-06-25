from src.utils.data_utils import symbolize, load_corpus
from src.utils.file_utils import read_yaml
from src.data.absa_data_manager import AbsaDataManager
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 80



def test_symbolize():
    sym_dict = read_yaml('tests/test_data/semeval14_symbol_dict.yml')
    test_sent = ['babish and cheap !'.split(),
                  'babish cheap and ! babish'.split()
                 ]

    test = symbolize(test_sent, sym_dict)
    answer = np.array([[  1,   5, 161,  21,   0],
                       [  1, 161,   5,  21,   1]])

    assert (test==answer).all()



def test_dm():

    test_sent = ['babish and cheap !'.split(),
                 'babish cheap and ! babish'.split(),
                 'hellow world ! babish'.split()
                 ]

    test_df = pd.DataFrame(data={'ASP': ['food', 'service', 'blahblahblah'],
                                 'CLS': [-1, 0, 1],
                                 'SENT': [' '.join(s) for s in test_sent]
                                 })

    dm = AbsaDataManager()

    dm.token2symbol(test_sent)

    _, batch = next(dm.batch_generator(test_df, 3))
    X, asp, lx, y = batch


    _X = np.array([[  1,   5, 696,  35,   0],
                   [  1, 696,   5,  35,   1],
                   [  1, 177,  35,   1,   0]])

    _asp = np.array([[347],
                    [227],
                    [  1]])

    _lx = None

    _y = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])

    assert (X==_X).all()
    assert (_asp==asp).all()
    assert _lx==lx
    assert (_y==y).all()