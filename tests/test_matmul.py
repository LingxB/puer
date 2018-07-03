import numpy as np
import tensorflow as tf
from src.utils.tf_utils import matmul_2_3



def test_matmul_2_3():
    X = tf.constant(np.random.normal(loc=0, scale=10, size=(3, 4, 5)).astype('int32')) # (3, 4, 5)
    _X = tf.transpose(X, [0, 2, 1]) # (3, 5, 4)
    assert _X.shape == (3, 5, 4)

    W = tf.constant(np.random.normal(loc=0, scale=1, size=(5, 5)).astype('int32')) # (5, 5)
    _W = tf.reshape(W, (1, W.shape[0], W.shape[1])) # (1, 5, 5)
    assert _W.shape == (1, 5, 5)

    _W = tf.tile(_W, [_X.shape[0], 1, 1])
    assert _W.shape == (3, 5, 5)

    out = tf.matmul(_W, _X)
    check = tf.matmul(W, _X[0,:,:])
    m23_out = matmul_2_3(W, _X)


    sess = tf.InteractiveSession()
    out_ = out.eval()
    check_ = check.eval()
    m23_out_ = m23_out.eval()
    sess.close()

    assert (out_[0,:,:] == check_).all()
    assert (out_==m23_out_).all()

