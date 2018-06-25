from src.utils.data_utils import symbolize
from src.utils.file_utils import load_symbod_dict_if_exists
import numpy as np





def test_symbolize():
    sym_dict = load_symbod_dict_if_exists('tests/test_data/semeval14_symbol_dict.yml')
    test_sent = ['babish and cheap !'.split(),
                  'babish cheap and ! babish'.split()
                 ]

    test = symbolize(test_sent, sym_dict)
    answer = np.array([[  1,   5, 161,  21,   0],
                       [  1, 161,   5,  21,   1]])

    assert (test==answer).all()
