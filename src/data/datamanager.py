"""
DataManager() object handles processes for manipulating data:

1. Load data from file
2. Symbolize
3. Bucketing
4. Padding
5. Batch generator

"""

import pandas as pd





class DataManager(object):

    def __init__(self):
        pass


    def load_data(self, file, **kwargs):
        _df = pd.read_csv(file, **kwargs)
        return _df


    def symbolize(self):
        pass


    def bucketing(self):
        pass


    def padding(self):
        pass


    def batch_generator(self):
        pass
