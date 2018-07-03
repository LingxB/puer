import tensorflow as tf


def matmul_2_3(tensor_2d, tensor_3d):
    """
    Matmul broadcasting 2d tensor to 3d tensor

    Parameters
    ----------
    tensor_2d : tensor
        i.e. a (5,5) tensor
    tensor_3d : tensor
        i.e. a (3, 5, 4)

    Returns
    -------
    tensor
        i.e. a (3, 5, 4) tensor

    """

    # reshape 2d to 3d

    left = tf.reshape(tensor_2d, (1, tensor_2d.shape[0], tensor_2d.shape[1]))
    left = tf.tile(left, [tf.shape(tensor_3d)[0], 1, 1])

    return tf.matmul(left, tensor_3d)


def seq_length(sequence):
    """
    Compute real sequence length on 3d tensor where 0 is used for padding. i.e. Tensor with shape (batch, N, d),
    sequence is padded with 0 vectors to form N as max_length of batch. This function computes the real length of each
    example in batch.

    Parameters
    ----------
    sequence : tensor
        3D tensor with shape (batch, N, d) where empty time steps are padded with 0 vector

    Returns
    -------
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

