import tensorflow as tf

from ..ops import optimizer as opt
from ..ops import sampler as sp
from ..network.registry import get_model
from .base_model import BaseModel
from .ird_model import IRDModel

"""
A Model designed for testing Noisy Adam on the toy x3 dataset.
"""

class X3Model(IRDModel):

    def __init__(self, config):
        super().__init__(config)


    def build_model(self):
        config = self.config

        # Define network parameters.
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.n_particles = tf.placeholder(tf.int32, [], name="n_particles")

        self.n_main = tf.constant(config.n_main, dtype=tf.int32, name="n_main")
        self.n_aux = tf.constant(config.n_aux, dtype=tf.int32, name="n_aux")
        self.n_timesteps = tf.constant(config.n_timesteps, dtype=tf.int32,
                name="n_timesteps")
        self.n_features = tf.constant(config.n_features, dtype=tf.int32,
                name="n_features")
        self.n_data = (config.n_main + config.n_aux) * config.n_timesteps

        # Define network inputs.
        self.inputs = tf.placeholder(tf.float32, [config.n_main,
            config.n_timesteps, 1], name="main_inputs")
        self.aux_inputs = tf.placeholder(tf.float32, [config.n_aux,
            config.n_timesteps, 1], name="aux_inputs") # Should be zero shape.
        self.sampler = sp.Sampler(self.config, self.n_data, self.n_particles)

        # Construct network.
        net = get_model(self.config.model_name)

        (self.main_outputs, self.aux_outputs, _, _,
                _, _, self.assert_group, self.param_dict
            ) = net(
                main_input_traj=self.inputs,
                aux_input_traj=self.aux_inputs,
                n_main=self.n_main,
                n_aux=self.n_aux,
                n_timesteps=self.n_timesteps,
                n_features=self.n_features,
                sampler=self.sampler,
                is_training=self.is_training,
                batch_norm=self.config.batch_norm,
                n_particles=self.n_particles)

        more_asserts = [
                tf.assert_equal(tf.shape(self.main_outputs),
                    (self.n_particles, self.n_main),
                    name="assert_aux_input_shape"),
                tf.assert_equal(tf.shape(self.aux_outputs),
                    (self.n_particles, self.n_aux),
                    name="assert_aux_output_shape"), ]

        # Define regression loss.
        self.targets = tf.placeholder(tf.float32,
                [config.n_main, 1], name="targets")
        targets_ = tf.tile(self.targets, [self.n_particles, 1],
                name="targets_tiled")

        more_asserts.append(tf.assert_equal(tf.shape(targets_),
            tf.shape(self.main_outputs)))

        with tf.control_dependencies([self.assert_group] + more_asserts):
                self.loss = tf.losses.mean_squared_error(targets_,
                        self.main_outputs)
