from src.utils import Logger, __fn__, load_corpus, load_embedding
from src.data import AbsaDataManager
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from copy import deepcopy
from time import time


logger = Logger(__fn__())

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

# Hyper parameters
hyparams = dict(
    random_state=None, # TODO: TEST FIXED RANDOM SEED
    batch_size=25,
    cell_num=300,
    layer_num=1,
    dropout_keep_prob=0.5,
    epsilon=0.01,
    momentum=0.9, # TODO: Momentum + AdaGrad = Adam?
    learning_rate=0.01, # AdaGrad initial
    lambta = 0.01  # L2
)

# Load data
dm = AbsaDataManager()

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')

# Input place holders
X = tf.placeholder(tf.int32, shape=(None, None), name='X')
asp = tf.placeholder(tf.int32, shape=(None, 1), name='asp')
y = tf.placeholder(tf.int32, shape=(None, 3), name='y')

dropout = tf.placeholder(tf.float32, shape=(1, ), name='dropout_keep')

# TODO: IMPLEMENT LX
# lx = tf.placeholder(tf.int32, shape=(None, None), name='lx')

# Embedding
with tf.name_scope('Embedding'):
    glove = tf.get_variable(name='glove',
                            shape=dm.emb.shape,
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(dm.emb.values),
                            trainable=True)
    pad = tf.get_variable(name='pad',
                          shape=(1, glove.shape[1].value),
                          dtype=tf.float32,
                          initializer=tf.zeros_initializer(),
                          trainable=False)
    unk = tf.get_variable(name='unk',
                          shape=(1, glove.shape[1].value),
                          dtype=tf.float32,
                          initializer=tf.random_uniform_initializer(minval=-hyparams['epsilon'],
                                                                    maxval=hyparams['epsilon'],
                                                                    seed=hyparams['random_state']),
                          trainable=True)
    embedding = tf.concat([pad, unk, glove], axis=0, name='embedding')

    X_ = tf.nn.embedding_lookup(embedding, X) # (batch, max_len, embedding_size)
    seq_len = length(X_)
    asp_ = tf.nn.embedding_lookup(embedding, asp) # (batch, 1, embedding_size)


with tf.name_scope('Encoder'):
    cell = rnn.BasicLSTMCell(hyparams['cell_num'])
    # cell = rnn.LSTMCell(hyparams['cell_num'])
    cells = [deepcopy(cell) for i in range(hyparams['layer_num'])]
    cell = rnn.MultiRNNCell(cells)

    H, (s,) = tf.nn.dynamic_rnn(cell=cell, inputs=X_, sequence_length=seq_len, dtype=tf.float32)




init = tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)
    _, first_batch = next(dm.batch_generator(train_df, batch_size=5))
    _X, _asp, _lx, _y = first_batch

    #X_lookup, s_len, asp_lookup = sess.run([X_, seq_len, asp_], feed_dict={X: _X, asp: _asp})

    H, s = sess.run([H, s], feed_dict={X: _X, asp: _asp})



