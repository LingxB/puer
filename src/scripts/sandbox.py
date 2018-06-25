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


dm = AbsaDataManager()


train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')


X = tf.placeholder(tf.int32, shape=(None, None), name='X')


pre_trained_embedding = tf.get_variable(name="pre_trained_embedding",
                                        shape=dm.emb.shape,
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(dm.emb.values),
                                        trainable=True)

pad_embedding = tf.get_variable(name='pad_embedding',
                                shape=(2, 300),
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer(),
                                trainable=False)

embedding = tf.concat([pad_embedding, pre_trained_embedding], axis=0, name='concat_embedding')


X_ = tf.nn.embedding_lookup(embedding, X)

seq_len = length(X_)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    _, first_batch = next(dm.batch_generator(train_df, batch_size=5))
    _X, _asp, _lx, _y = first_batch
    lookup, s_len = sess.run([X_, seq_len], feed_dict={X: _X})



