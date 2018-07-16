import tensorflow as tf
from src.data import AbsaDataManager
from src.utils import Logger, __fn__, load_corpus #, matmul_2_3, seq_length, get_envar, read_config, get_timestamp, mkdir



logger = Logger(__fn__())


dm = AbsaDataManager()

dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')




sess = tf.Session()

saver = tf.train.import_meta_graph('models/20180715_193759/AT-LSTM.meta')
saver.restore(sess, tf.train.latest_checkpoint('models/20180715_193759'))

graph = tf.get_default_graph()
all_ops = [op.name for op in graph.get_operations()]


X = graph.get_tensor_by_name('X:0')
asp = graph.get_tensor_by_name('asp:0')
y = graph.get_tensor_by_name('y:0')

pred = graph.get_tensor_by_name('Output/Softmax:0')
accuarcy = graph.get_tensor_by_name('Evaluation/Mean:0')

batch_generator = dm.batch_generator(test_df, 25)

_, batch = next(batch_generator)
_X, _asp, _lx, _y = batch

pred_, accuarcy_ = sess.run([pred, accuarcy], feed_dict={X: _X, asp: _asp, y: _y})


sess.close()