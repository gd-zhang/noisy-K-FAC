from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import weight_blocks as wb

from collections import OrderedDict

# Names for various approximations that can be requested for weight matrix.
APPROX_KRONECKER_NAME = "kron"
APPROX_DIAGONAL_NAME = "diagonal"

_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: wb.MVGBlock,
    APPROX_DIAGONAL_NAME: wb.FFGBlock,
}


class Sampler(object):
    def __init__(self, config, n_data, particles):
        self.config = config
        self._particles = particles
        self._n_data = n_data
        self.blocks = OrderedDict()

    def get_block(self):
        return self.blocks.values()

    def get_params(self, idx):
        if idx not in self.blocks:
            raise ValueError("Unvalid query: {}".format(idx))
        return self.blocks[idx].params()

    def sample(self, idx):
        if idx not in self.blocks:
            raise ValueError("Unvalid query: {}".format(idx))
        return self.blocks[idx].sample(self._particles)

    def register_block(self, idx, shape):
        if idx in self.blocks:
            raise ValueError("Duplicate registration: {}".format(idx))

        block_type = _APPROX_TO_BLOCK_TYPES[self.config.fisher_approx]
        self.blocks[idx] = block_type(idx, shape, self.config.kl/self._n_data, self.config.eta)

    def update(self, blocks):
        block_list = self.get_block()
        update_op = [wb.update(fb) for wb, fb in zip(block_list, blocks)]
        return tf.group(*update_op)




