import tensorflow as tf


def _append_homog(tensor):
    """
    Adds homogeneous input as last element of last dimension of tensor.
    """
    rank = len(tensor.shape.as_list())
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
    ones = tf.ones(shape, dtype=tensor.dtype)
    return tf.concat([tensor, ones], axis=rank - 1)


def dense(inputs, weights, batch_norm, is_training, particles=1):
    """
    Automatically adds bias units.
    """
    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    inputs = tf.reshape(inputs, [particles, -1, n_in])
    preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)

    return preactivations, activations

def dup_for_particles(inputs, particles):
    """
    Prepend a new dimension 0 where inputs is duplicated `particles` times.
    """
    rank = len(inputs.shape.as_list())
    ones = tf.ones([particles] + [1]*rank)
    return tf.identity(ones * inputs, name="psuedotile")

def dense_feat(inputs, weights, batch_norm, is_training, particles,):
    """
    Like dense(), but preserves a final n_features dimensions for IRD.
    """
    inputs = _append_homog(inputs)
    # inputs = dup_for_particles(inputs, particles)
    preactivations = tf.matmul(inputs, weights)
    # preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)

    return preactivations, activations


def conv2d(inputs, weights, filter_shape, batch_norm, is_training,
           particles=1, strides=(1, 1, 1, 1), padding="SAME"):
    filter_height, filter_width, in_channels, out_channels = filter_shape

    def f1():
        weight = tf.reshape(weights[0, :-1, :], filter_shape)
        bias = tf.squeeze(weights[0, -1, :])
        conv = tf.nn.conv2d(inputs, filter=weight,
                            strides=strides, padding=padding)
        return tf.nn.bias_add(conv, bias)

    def f2():
        patches = tf.extract_image_patches(
            inputs,
            ksizes=[1, filter_height, filter_width, 1],
            strides=strides,
            rates=[1, 1, 1, 1],
            padding=padding)

        patches = _append_homog(patches)
        pb, h_out, w_out, flatten_size = patches.shape.as_list()
        patches = tf.reshape(patches, [particles, -1, flatten_size])
        preactivations = tf.matmul(patches, weights)
        preactivations = tf.reshape(preactivations, [-1, h_out, w_out, out_channels])
        return preactivations

    preactivations = tf.cond(tf.equal(particles, 1), f1, f2)
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = tf.nn.relu(bn)
    else:
        activations = tf.nn.relu(preactivations)
    return preactivations, activations
