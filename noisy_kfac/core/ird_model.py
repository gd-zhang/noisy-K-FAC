import tensorflow as tf

from ..ops import optimizer as opt
from ..ops import sampler as sp
from ..network.registry import get_model
from .base_model import BaseModel

"""
Specialized version of Model for IRD tasks.
"""

class IRDModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.cov_update_op = None
        self.inv_update_op = None

        self.build_model()
        self.init_optim()
        self.init_saver()


    def build_model(self):
        # Define network parameters.
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.n_particles = tf.placeholder(tf.int32, name="n_particles")

        # TODO: Is it still realistic to have constant n_main and n_aux?
        # I suppose that I will set these deterministically at first,
        # and probably remove these requirements later.
        self.n_main = tf.constant(config.n_main, dtype=tf.uint32, name="n_main")
        self.n_aux = tf.constant(config.n_aux, dtype=tf.uint32, name="n_aux")
        self.n_timesteps = tf.constant(config.n_timesteps, dtype=tf.uint32,
                name="n_timesteps")
        self.n_features = tf.constant(config.n_features, dtype=tf.uint32,
                name="n_features")

        # Calculate the umber of inputs coming into the BNN per batch. Important
        # for sampling Fisher blocks and gradient updates.
        self.n_data = (config.n_main + config.n_aux) * config.n_timesteps

        # Define network inputs.
        self.main_inputs = tf.placeholder(tf.float32, [config.n_main,
            config.n_timesteps], name="main_inputs")
        self.aux_inputs = tf.placeholder(tf.float32, [config.n_aux,
            config.n_timesteps], name="aux_inputs")
        self.sampler = sp.Sampler(self.config, self.n_data, self.n_particles)

        # Construct network.
        net = get_model(self.config.model_name)

        (self.main_outputs, self.aux_outputs, self.aux_outputs_lse, self.ll_sep,
                self.ll, self.l2_loss, self.assert_group, self.param_dict
            ) = net(self.inputs, self.aux_inputs,
                self.sampler, self.is_training,
                self.config.batch_norm, self.n_particles)

        self.loss = tf.identity(-self.ll, name="loss")

        @property
        def trainable_variables(self):
            # For Noisy Adam, take the actual samples of w from our network.
            return list(self.param_dict.keys())


        def init_optim(self):
            self.optim = opt.AdamOptimizer(
                var_list=self.trainable_variables,
                param_dict = self.param_dict,
                learning_rate=self.config.learning_rate,
                cov_ema_decay=self.config_ema_decay,
                damping=self.config.damping,
                )

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optim.minimize(self.loss,
                        global_step=self.global_step_tensor, name="IRD_min")

        def get_loss_and_aux(self, sess, feed_dict=None):
            loss = sess.run([self.loss], feed_dict=feed_dict)
            log_aux = {}
            return loss, log_aux
