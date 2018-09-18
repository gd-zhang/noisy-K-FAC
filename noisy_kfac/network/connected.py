import tensorflow as tf
import numpy as np

from ..core import MODE_REGRESSION, MODE_IRD
from ..misc.layers import dense
from .registry import register_model

def FullyConnected(inputs, aux_inputs, sampler, is_training, batch_norm,
        layer_collection, particles, out_sizes, ird=False):
    """
    out_sizes (list[int]): A nonempty list of layer output sizes. For example,
        `[10, 1]` describes a fully connected network with a scalar output
        and a single hidden layer of size 10. Input size is inferred from
        `inputs`.
    """
    def FCBlock(inputs, aux_inputs, out_channel, layer_idx, ignore_activation=False):
        in_channel = inputs.shape.as_list()[-1]
        sampler.register_block(layer_idx, (in_channel, out_channel))
        weights = sampler.sample(layer_idx)
        l2_loss = 0.5 * tf.reduce_sum(weights ** 2)

        # True outputs
        pre, act = dense(inputs, weights, batch_norm, is_training,
                particles)
        layer_collection.register_fully_connected(
                sampler.get_params(layer_idx), inputs, pre)
        output = pre if ignore_activation else act

        # Aux outputs
        pre, act = dense(aux_inputs, weights, batch_norm, is_training,
                particles)
        aux_output = pre if ignore_activation else act

        return output, aux_output, l2_loss

    inputs = tf.tile(inputs, [particles, 1])
    aux_inputs = tf.tile(aux_inputs, [particles, 1])
    prev_inputs, prev_aux_inputs, l2_loss = inputs, aux_inputs, 0.

    for i, out_size in enumerate(out_sizes):
        # The final output is unactivated.
        ignore_activation = (i == len(out_sizes) - 1)
        prev_inputs, prev_aux_inputs, loss = FCBlock(prev_inputs, prev_aux_inputs,
                out_size, i, ignore_activation)
        l2_loss += loss
    pred, aux_pred = prev_inputs, prev_aux_inputs
    layer_collection.register_normal_predictive_distribution(pred,
            name="prediction")

    return pred, aux_pred, l2_loss, MODE_REGRESSION if not ird else MODE_IRD

def _build_fc(out_sizes):
    def fc(*args, **kwargs):
        return FullyConnected(*args, out_sizes=out_sizes, **kwargs)
    return fc

def _build_fc_ird(out_sizes):
    def fc(*args, **kwargs):
        return FullyConnected(*args, out_sizes=out_sizes, ird=True, **kwargs)
    return fc

register_model("fc1_10")(_build_fc([10, 1]))
register_model("fc1_20")(_build_fc([20, 1]))
register_model("fc2_10")(_build_fc([10, 10, 1]))
register_model("fc2_20")(_build_fc([20, 20, 1]))

register_model("fc1_1_ird")(_build_fc_ird([1, 1]))
register_model("fc1_5_ird")(_build_fc_ird([5, 1]))
register_model("fc1_10_ird")(_build_fc_ird([10, 1]))
register_model("fc1_20_ird")(_build_fc_ird([20, 1]))
register_model("fc2_10_ird")(_build_fc_ird([10, 10, 1]))
register_model("fc2_20_ird")(_build_fc_ird([20, 20, 1]))
