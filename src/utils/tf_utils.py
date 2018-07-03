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
    left = tf.tile(left, [tensor_3d.shape[0], 1, 1])

    return tf.matmul(left, tensor_3d)

