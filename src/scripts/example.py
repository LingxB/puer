from src.data import AbsaDataManager
from src.model import Model
from src.utils.file_utils import read_config


dm = AbsaDataManager()



configs = read_config()


model = Model(configs.params)


model.train()


model.save()






model.load()

model.test()

model.test_with_new_lexicon()



produce_report()