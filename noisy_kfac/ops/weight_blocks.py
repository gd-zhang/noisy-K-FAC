from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc


def _compute_pi_tracenorm(left_cov, right_cov):
    left_norm = tf.trace(left_cov) * right_cov.shape.as_list()[0]
    right_norm = tf.trace(right_cov) * left_cov.shape.as_list()[0]
    return tf.sqrt(left_norm / right_norm)


class WeightBlock(object):
    def __init__(self, idx, shape, coeff, eta):
        self._shape = shape
        self._n_in = np.prod(shape[:-1]) + 1
        self._n_out = shape[-1]
        self._coeff = coeff
        self._eta = eta
        self.idx = idx

        self._build_weights(idx)

    def _build_weights(self, idx):
        self._weight = tf.get_variable(
            'train_w_'+str(idx)+'_weight',
            shape=self._shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        self._bias = tf.get_variable(
            'train_w_'+str(idx)+'_bias',
            shape=[self._n_out],
            initializer=tf.constant_initializer(0.),
            trainable=True
        )

    def params(self):
        return (self._weight, self._bias)

    @property
    def _mean(self):
        weight = tf.reshape(self._weight, (self._n_in-1, self._n_out))
        bias = tf.expand_dims(self._bias, 0)
        return tf.concat([weight, bias], 0, name="mean")

    @abc.abstractmethod
    def sample(self, particles):
        pass

    @abc.abstractmethod
    def update(self, block):
        pass


class FFGBlock(WeightBlock):
    def __init__(self, idx, shape, coeff, eta):
        super(FFGBlock, self).__init__(idx, shape, coeff, eta)
        self._std = tf.get_variable(
            'train_w_'+str(idx)+'_std',
            shape=[self._n_in, self._n_out],
            initializer=tf.constant_initializer(1e-5),
            trainable=False
        )

    def sample(self, particles):
        mean = self._mean
        out_mean = tf.tile(tf.expand_dims(mean, 0), [particles, 1, 1])
        rand = tf.random_normal(shape=tf.shape(out_mean))
        std = self._std
        out_rand = std * rand
        return out_mean + out_rand

    def update(self, block):
        # Is self._coeff == damping_int? No, but almost. It's lambda/N. It still
        # needs to be divided by _eta to get damping_int.
        # Nice, so we have variance = (f + damping_int)^-1 here. Assuming that
        # `block._factor.get_cov() == f` here.
        # Nice, so we are assigning self._std with sqrt(var). Correcto.
        # What is get_cov()? Why is this equal to `f`?
        # Well, I guess we can just suppose that get_cov is the fisher matrix
        # provided from the FisherFactor class. OK.
        #
        # OK, so how do I use update_op?
        # Sampler.update() definitely uses it.
        # It looks convoluted though.
        variance = 1 / (block._factor.get_cov() + self._coeff / self._eta)
        update_op = self._std.assign(tf.sqrt(self._coeff * variance))
        return update_op

class FFG_IRDBlock(WeightBlock):
    def __init__(self, idx, shape, coeff, eta):
        super().__init__(idx, shape, coeff, eta)
        self._f = tf.get_variable(
                'train_w_'+str(idx)+'f_diag',
                shape=[self._n_in, self._n_out],
                initializer=tf.constant_initializer(1e-5),
                trainable=False
        )

        # self._coeff is preset to (lambda/N)
        self.damping_int = tf.identity(self._coeff / self._eta,
                name="damping_int")
        pre_variance = 1 / (self._f + self.damping_int)
        self._std = tf.sqrt(self._coeff * pre_variance, name="stddev")

    def params(self):
        """
        Returns the two params important for NoisyAdam updating --
        _mean and _f.

        Note that in all other WeightBlock classes, params() means something
        different.
        """
        # Dirty overloading here. Because other classes return other params.
        return self._mean, self._f

    def update(self, block):
        # Sanity check:
        raise NotImplementedError("IRD + NoisyAdam shouldn't use this update!")

    def sample(self, particles):
        mean = self._mean
        out_mean = tf.tile(tf.expand_dims(mean, 0), [particles, 1, 1])
        rand = tf.random_normal(shape=tf.shape(out_mean))
        std = self._std
        out_rand = tf.identity(std * rand, name="sample_noise")
        return tf.identity(out_mean + out_rand,
                name="weight_samples"+str(self.idx))


class MVGBlock(WeightBlock):
    def __init__(self, idx, shape, coeff, eta):
        super(MVGBlock, self).__init__(idx, shape, coeff, eta)
        self._u_c = tf.get_variable(
            'train_w_'+str(idx)+'_u_c',
            initializer=1e-3 * tf.eye(self._n_in),
            trainable=False
        )
        self._v_c = tf.get_variable(
            'train_w_'+str(idx)+'_v_c',
            initializer=1e-3 * tf.eye(self._n_out),
            trainable=False
        )

    def sample(self, particles):
        mean = self._mean
        out_mean = tf.tile(tf.expand_dims(mean, 0), [particles, 1, 1])
        rand = tf.random_normal(shape=tf.shape(out_mean))
        u_c = tf.tile(tf.expand_dims(self._u_c, 0), [particles, 1, 1])
        v_c = tf.tile(tf.expand_dims(self._v_c, 0), [particles, 1, 1])
        out_rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))

        return out_mean + out_rand

    def update(self, block):
        input_factor = block._input_factor
        output_factor = block._output_factor
        pi = _compute_pi_tracenorm(input_factor.get_cov(), output_factor.get_cov())

        coeff = self._coeff / block._renorm_coeff
        coeff = coeff ** 0.5
        damping = coeff / (self._eta ** 0.5)

        ue, uv = tf.self_adjoint_eig(
            input_factor.get_cov() / pi + damping * tf.eye(self._u_c.shape.as_list()[0]))
        ve, vv = tf.self_adjoint_eig(
            output_factor.get_cov() * pi + damping * tf.eye(self._v_c.shape.as_list()[0]))

        ue = coeff / tf.maximum(ue, damping)
        new_uc = uv * ue ** 0.5

        ve = coeff / tf.maximum(ve, damping)
        new_vc = vv * ve ** 0.5

        updates_op = [self._u_c.assign(new_uc), self._v_c.assign(new_vc)]
        return tf.group(*updates_op)
