from src.utils import Logger, __fn__, load_corpus, matmul_2_3, seq_length, get_envar, read_config, get_timestamp, mkdir
from src.data import AbsaDataManager
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from copy import deepcopy
from time import time
import pandas as pd

logger = Logger(__fn__())

# Configs
# -------
base_configs = read_config(get_envar('CONFIG_PATH') + '/' + get_envar('BASE_CONFIG'), obj_view=True)
wdir = base_configs.model.savepath + get_timestamp() + '/'


# Hyper parameters
# ----------------
exp_num = 'exp_' + input('exp_?')
exp_configs = read_config(base_configs.exp_configs.path, obj_view=False)[exp_num]
hyparams = exp_configs['hyperparams']
description = exp_configs['description']

# Load data
# ---------
dm = AbsaDataManager()

train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')


# Input place holders
# -------------------
X = tf.placeholder(tf.int32, shape=(None, None), name='X')
asp = tf.placeholder(tf.int32, shape=(None, 1), name='asp')
y = tf.placeholder(tf.int32, shape=(None, 3), name='y')
lx = tf.placeholder(tf.float32, shape=(None, None, None), name='lx')

dropout_keep = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep')


# Initializer
# -----------
initializer = tf.random_uniform_initializer(minval=hyparams['minval'],
                                            maxval=hyparams['maxval'],
                                            seed=hyparams['seed'])

# Embedding
# ---------
with tf.name_scope('Embedding'):
    glove = tf.get_variable(name='glove',
                            shape=dm.emb.shape,
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(dm.emb.values),
                            )
    pad = tf.get_variable(name='pad',
                          shape=(1, glove.shape[1]),
                          dtype=tf.float32,
                          initializer=tf.zeros_initializer(),
                          trainable=False)
    unk = tf.get_variable(name='unk',
                          shape=(1, glove.shape[1]),
                          dtype=tf.float32,
                          initializer=initializer,
                          )
    embedding = tf.concat([pad, unk, glove], axis=0, name='embedding')

    X_ = tf.nn.embedding_lookup(embedding, X) # (batch, N, d)
    seq_len = seq_length(X_)
    asp_ = tf.nn.embedding_lookup(embedding, asp) # (batch, 1, da)


# Encoder
# -------
with tf.name_scope('Encoder'):
    # cell = rnn.BasicLSTMCell(hyparams['cell_num'])
    cell = rnn.LSTMCell(hyparams['cell_num'], initializer=initializer)
    cells = [deepcopy(cell) for i in range(hyparams['layer_num'])]
    cell = rnn.MultiRNNCell(cells)

    H, (s,) = tf.nn.dynamic_rnn(cell=cell, inputs=X_, sequence_length=seq_len, dtype=tf.float32) # (batch, N, d)
    hN = s.h # (batch, d)
    assert H.shape.as_list() == tf.TensorShape([X_.shape[0], X_.shape[1], hyparams['cell_num']]).as_list()


# Attention
# ---------
with tf.name_scope('Attention'):
    H_T = tf.transpose(H, [0, 2, 1]) # (batch, d, N)
    Wh = tf.get_variable('Wh', shape=(hyparams['cell_num'], hyparams['cell_num']), dtype=tf.float32, initializer=initializer) # (d, d)
    WhH = matmul_2_3(Wh, H_T) # (batch, d, N)
    assert WhH.shape.as_list() == H_T.shape.as_list()

    VaeN = tf.tile(asp_, [1, tf.shape(X_)[1], 1]) # (batch, N, da), da==d in this setting
    assert VaeN.shape.as_list() == X_.shape.as_list()
    VaeN_T = tf.transpose(VaeN, [0, 2, 1]) # (batch, da, N)
    Wv = tf.get_variable('Wv', shape=(asp_.shape[2], asp_.shape[2]), dtype=tf.float32, initializer=initializer) # (da, da)
    WvVaeN = matmul_2_3(Wv, VaeN_T) # (batch, da, N)
    assert WvVaeN.shape.as_list() == VaeN_T.shape.as_list()

    M = tf.tanh(tf.concat([WhH, WvVaeN], axis=1)) # (batch, d+da, N)

    w = tf.get_variable('w', shape=(X_.shape[2]+asp_.shape[2], 1), dtype=tf.float32, initializer=initializer) # (d+da, 1)
    w_T = tf.transpose(w) # (1, d+da)
    alpha = tf.nn.softmax(matmul_2_3(w_T, M), name='ALPHA') # (batch, 1, N)
    assert alpha.shape.as_list() == tf.TensorShape([X_.shape[0], 1, M.shape[2]]).as_list()
    #alpha = tf.reshape(alpha, (tf.shape(alpha)[0], tf.shape(alpha)[2])) # (batch, N)

    _r = tf.matmul(alpha, H) # (batch, 1, d)
    r = tf.squeeze(_r, 1) # (batch, d)
    assert r.shape.as_list() == tf.TensorShape([X_.shape[0], H.shape[2]]).as_list()

    Wp = tf.get_variable('Wp', shape=(r.shape[1], r.shape[1]), dtype=tf.float32, initializer=initializer) # (d, d)
    Wx = tf.get_variable('Wx', shape=(hN.shape[1], hN.shape[1]), dtype=tf.float32, initializer=initializer) # (d, d)
    h_star = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hN, Wx))
    h_star = tf.nn.dropout(h_star, dropout_keep) # 0.5 dropout on h_star was found in author's code
    assert h_star.shape.as_list() == tf.TensorShape([H.shape[0], H.shape[2]]).as_list()


# Output Layer
# ------------
with tf.name_scope('Output'):
    logits = tf.layers.dense(h_star, 3, kernel_initializer=initializer, name='s')
    pred = tf.nn.softmax(logits, name='PRED')


# Loss
# ----
with tf.name_scope('Loss'):
    # TODO: Check loss, use reduce_sum instead of reduce_mean
    # TODO: Check L2, current implenmentation loss is not normalized by batch_size
    # TODO: Check embedding params, current implementation includes embedding params in L2 regularization

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    reg_params = [p for p in tf.trainable_variables() if p.name not in {'glove:0', 'unk:0'}]
    # regularizer = hyparams['lambda'] * tf.add_n([tf.nn.l2_loss(p) for p in reg_params])
    regularizer = tf.multiply(hyparams['lambda'], tf.add_n([tf.nn.l2_loss(p) for p in reg_params]), name='REGL')
    # regularizer = hyparams['lambda'] * tf.add_n([tf.nn.l2_loss(p) for p in tf.trainable_variables()])
    loss = tf.add(cross_entropy, regularizer, name='LOSS') # cross_entropy + regularizer

    # reg_params = [p for p in tf.trainable_variables() if p.name not in {'glove:0', 'unk:0'}]
    # regularizer = hyparams['lambda'] * tf.add_n([tf.nn.l2_loss(p) for p in reg_params])
    # cross_entropy_by_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    # cross_entropy = tf.reduce_mean(cross_entropy_by_example)
    # loss = tf.reduce_mean(cross_entropy_by_example + regularizer) # This is wrong


# Train Op
# --------
with tf.name_scope('TrainOp'):
    if hyparams['optimizer'] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(hyparams['learning_rate'])
    elif hyparams['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=hyparams['learning_rate'], beta1=hyparams['momentum'])
    elif hyparams['optimizer'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=hyparams['learning_rate'])
    elif hyparams['optimizer'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=hyparams['learning_rate'], momentum=hyparams['momentum'])
    elif hyparams['optimizer'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=hyparams['learning_rate'], momentum=hyparams['momentum'])
    else:
        raise NotImplementedError('Invalid optimizer in hyperparams')

    train_op = optimizer.minimize(loss, name='TRAIN_OP')


# Evaluation during training
with tf.name_scope('Evaluation'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='ACC3')


# Test random seed
# print(hyparams)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     _, first_batch = next(dm.batch_generator(train_df, batch_size=hyparams['batch_size']))
#     _X, _asp, _lx, _y = first_batch
#     print(sess.run(loss, feed_dict={X: _X, asp: _asp, y: _y})) # reduce_sum, random_state=4, loss=27.483023


# Run
# ---
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    logger.info('---- {} training started ----'.format(exp_num))
    logger.info('description: {}'.format(description))
    logger.info('hyperparams: {}'.format(hyparams))
    _, test_batch = next(dm.batch_generator(test_df, batch_size=-1))
    X_test, asp_test, lx_test, y_test = test_batch
    for epoch in range(hyparams['epochs']):
        batch_generator = dm.batch_generator(train_df, batch_size=hyparams['batch_size'], shuffle=True)
        for i, (_, batch) in enumerate(batch_generator):
            _X, _asp, _lx, _y = batch

            _cross_entropy, _regularizer, _loss, _accuarcy, _ = \
                sess.run([cross_entropy,
                          regularizer,
                          loss,
                          accuracy,
                          train_op],
                         feed_dict={X: _X,
                                    asp: _asp,
                                    y: _y,
                                    dropout_keep: hyparams['dropout_keep_prob']
                                    })

            logger.debug('epoch {epoch:03d}/{epochs:03d} \t'
                         'batch {i:02d}/{n_batches:02d} \t'
                         'error={cross_entropy:4.4f} \t'
                         'l2={l2:4.4f} \t'
                         'loss={loss:4.4f} \t'
                         'train_acc/3={acc:.2%}'
                         .format(epoch=epoch+1, epochs=hyparams['epochs'], i=i, n_batches=dm.n_batches,
                                 cross_entropy=_cross_entropy, l2=_regularizer, loss=_loss, acc=_accuarcy,
                                 ))

        test_accuarcy = sess.run(accuracy, feed_dict={X: X_test, asp: asp_test, y: y_test})
        logger.info('** Epoch {epoch:03d}/{epochs:03d} '
                    'Test accuarcy: {ta:.2%} **'.format(epoch=epoch, epochs=hyparams['epochs'], ta=test_accuarcy))

    logger.info('---- {} training ended ----'.format(exp_num))
    logger.info('Saving model...')
    mkdir(wdir)
    saver.save(sess, wdir + 'AT-LSTM')
    logger.info("Model saved to '{}'".format(wdir))


    # _, first_batch = next(dm.batch_generator(train_df, batch_size=hyparams['batch_size']))
    # _X, _asp, _lx, _y = first_batch
    #
    # sess.run()


    #X_lookup, s_len, asp_lookup = sess.run([X_, seq_len, asp_], feed_dict={X: _X, asp: _asp})

    # H, s, _H = sess.run([H, s, _H], feed_dict={X: _X, asp: _asp})


    # H, Wh, WhH = sess.run([H, Wh, WhH], feed_dict={X: _X, asp: _asp})
#
#
#
# _H = H.transpose([2, 1, 0])
# __H = _H.reshape(300, -1)



