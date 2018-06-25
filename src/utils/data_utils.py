import pandas as pd
from collections import Counter
from tensorflow.python.keras.preprocessing.sequence import pad_sequences




def load_corpus(file, **kwargs):
    if isinstance(file, str):
        _df = pd.read_csv(file, **kwargs)
    elif isinstance(file, list):
        _df = pd.concat([pd.read_csv(f, **kwargs) for f in file], ignore_index=True)
    else:
        raise AttributeError('File type not valid, path or list of paths only.')
    return _df


def create_symbol_dict(corpus, start_idx=2):
    """

    Parameters
    ----------
    corpus : ``list`` or ``pandas.Series`` or ``numpy.ndarray``
        An iterable (corpus) contains sentences and tokens. i.e. [['hello','world'], ['w3','w4'...], ['w8','w9',...], ...]
    start_idx : ``int``
        Start index symbol, 0 reserved for padding, 1 reserved for <UNK>
    Returns
    -------
    ``dict``
        Word to symbol dictionary
    ``Counter``
        ``collections.Counter`` object with corpus word counts
    """
    c = Counter()
    for s in corpus:
        c.update([w for w in s])

    d = {w: i + start_idx for i, (w, _) in enumerate(c.most_common())}
    return d, c


def symbolize(corpus, symbol_dict, unk=2, **kwargs):
    return pad_sequences([[symbol_dict.get(w, unk) for w in s] for s in corpus], padding='post', **kwargs)