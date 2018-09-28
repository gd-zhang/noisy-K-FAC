import tensorflow as tf

from ..ops import optimizer as opt
from ..ops import layer_collection as lc
from ..ops import sampler as sp
from ..network.registry import get_model
from .base_model import BaseModel

"""
Specialized version of Model for IRD tasks.
"""

class IRDModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.layer_collection = lc.LayerCollection()
        self.cov_update_op = None
        self.inv_update_op = None

        self.build_model()
        self.init_optim()
        self.init_saver()


    def build_model(self):
        # Define network parameters.
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.n_particles = tf.placeholder(tf.int32, name="n_particles")
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
        self.main_input = tf.placeholder(tf.float32, [config.n_main,
            config.n_timesteps], name="main_input")
        self.aux_input = tf.placeholder(tf.float32, [config.n_aux,
            config.n_timesteps], name="aux_input")
        self.sampler = sp.Sampler(self.config, self.n_data, self.n_particles)

        # Construct network.
        net = get_model(self.config.model_name)

        (self.main_output, self.aux_output, self.aux_output_lse, self.ll_sep,
                self.ll, self.l2_loss, self.assert_group
            ) = net(self.inputs, self.aux_inputs,
                self.sampler, self.is_training,
                self.config.batch_norm, self.layer_collection,
                self.n_particles)

        # For compatibility with the rest of the package.
        self.acc = tf.constant(0, name="acc")

        coeff = self.config.kl / (self.n_data * self.config.eta)
        self.total_loss = tf.identity(self.loss + coeff * self.l2_loss,
                name="total_loss")


