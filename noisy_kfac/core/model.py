import tensorflow as tf

from ..ops import optimizer as opt
from ..ops import layer_collection as lc
from ..ops import sampler as sp
from ..network.registry import get_model
from .base_model import BaseModel
from . import MODE_CLASSIFICATION, MODE_REGRESSION, MODE_IRD

class Model(BaseModel):
    def __init__(self, config, input_dim, n_data):
        super().__init__(config)
        self.layer_collection = lc.LayerCollection()
        self.input_dim = input_dim
        self.n_data = n_data
        self.cov_update_op = None
        self.inv_update_op = None

        self.build_model()
        self.init_optim()
        self.init_saver()

    @property
    def trainable_variables(self):
        # note: we don't train the params of BN
        vars = []
        for var in tf.trainable_variables():
            if "w" in var.name:
                vars.append(var)
        return vars

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.input_dim,
                name="inputs")
        self.aux_inputs = tf.placeholder(tf.float32, [None] + self.input_dim,
                name="aux_inputs")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.n_particles = tf.placeholder(tf.int32, name="n_particles")

        net = get_model(self.config.model_name)

        self.sampler = sp.Sampler(self.config, self.n_data, self.n_particles)
        outputs, aux_outputs, l2_loss, mode = net(self.inputs, self.aux_inputs,
                              self.sampler, self.is_training,
                              self.config.batch_norm, self.layer_collection,
                              self.n_particles)

        self.aux_outputs = aux_outputs
        self.outputs = outputs
        self.mode = mode


        if mode == MODE_REGRESSION:
            self.targets = tf.placeholder(tf.float32, [None])
            targets_ = tf.tile(self.targets, [self.n_particles])
            self.loss = tf.losses.mean_squared_error(targets_,
                    tf.squeeze(outputs, 1))
            self.acc = self.loss
        elif mode == MODE_IRD:
            self.targets = None
            self.aux_lse = tf.reduce_logsumexp(self.aux_outputs, name="aux_lse")
            self.loss = -tf.reduce_mean(tf.subtract(self.outputs, self.aux_lse, name="loss"))
            self.acc = self.loss
        elif mode == MODE_CLASSIFICATION:
            self.targets = tf.placeholder(tf.int64, [None])
            targets_ = tf.tile(self.targets, [self.n_particles])
            logits_ = tf.reduce_mean(
                tf.reshape(outputs, [self.n_particles, -1, tf.shape(outputs)[-1]]
                    ), 0)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(
                self.targets, tf.argmax(logits_, axis=1)), dtype=tf.float32))
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets_, logits=outputs))
        else:
            raise TypeError("Illegal mode: {}".format(mode))


        coeff = self.config.kl / (self.n_data * self.config.eta)
        self.total_loss = self.loss + coeff * l2_loss

    def init_optim(self):
        self.optim = opt.KFACOptimizer(var_list=self.trainable_variables,
                                       learning_rate=self.config.learning_rate,
                                       cov_ema_decay=self.config.cov_ema_decay,
                                       damping=self.config.damping,
                                       layer_collection=self.layer_collection,
                                       norm_constraint=tf.train.exponential_decay(self.config.kl_clip,
                                                                                  self.global_step_tensor,
                                                                                  390, 0.95, staircase=True),
                                       momentum=self.config.momentum)

        self.cov_update_op = self.optim.cov_update_op
        self.inv_update_op = self.optim.inv_update_op

        with tf.control_dependencies([self.inv_update_op]):
            self.var_update_op = self.sampler.update(self.layer_collection.get_blocks())

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

