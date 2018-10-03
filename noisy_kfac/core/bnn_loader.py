from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ..misc.utils import get_logger, get_args, makedirs
from ..misc.config import process_config
from ..misc.data_loader import load_pytorch
from .kfac_model import KFACModel
from .x3_model import X3Model
from .ird_model import IRDModel
from .kfac_train import KFACTrainer
from .adam_train import NoisyAdamTrainer

_INPUT_DIM = {
    'fmnist': [784],
    'mnist': [784],
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3],
    'x3': [1],
}


class BNNLoader():
    logger = None

    def __init__(self, input_size=None, config_name="x3/kfac_ird.json",
            train_loader=None, test_loader=None):

        self.sess = tf.Session()

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(parent_dir, '../../')
        configpath = os.path.join(root_dir, 'configs/', config_name)
        self.config = config = process_config(configpath)

        # set logger
        path1 = os.path.join(root_dir, 'noisy_kfac/core/model.py')
        path2 = os.path.join(root_dir, 'noisy_kfac/core/train.py')
        # Logger is shared between each instance. This prevents
        # duplicate messages.
        BNNLoader.logger = BNNLoader.logger or get_logger('log',
                            logpath=config.summary_dir+'/',
                            filepath=os.path.abspath(__file__),)
                            # package_files=[path1, path2],)

        self.logger.info(self)
        self.logger.info(str(config.items()))

        # load data
        train_loader_cfg, test_loader_cfg = load_pytorch(config)

        self.train_loader = train_loader
        if self.train_loader is None:
            self.train_loader = train_loader_cfg

        self.test_loader = test_loader
        if self.test_loader is None:
            self.test_loader = test_loader_cfg

        assert self.train_loader is not None
        assert self.test_loader is not None

        # define computational graph
        self.tag = tag = config.get("tag")
        if tag == "x3ird":  # Using X3 dataset to test NoisyAdam-IRD model.
            config["n_main"] = config.batch_size
            config["n_aux"] = 0
            config["n_timesteps"] = 1
            config["n_features"] = 1
            config["fisher_approx"] = "ird_diag"
            self.model = X3Model(config)
            assert self.model.n_data == self.config.batch_size
            self.trainer = NoisyAdamTrainer(self.sess, self.model,
                    self.train_loader, self.test_loader, config,
                    self.logger)
        else:
            input_size = input_size or _INPUT_DIM[self.config.dataset]
            self.model = KFACModel(config, input_size,
                    len(self.train_loader))
            self.trainer = Trainer(self.sess, self.model, self.train_loader,
                    self.test_loader, config, self.logger)


    def train(self, aux_inputs):
        if self.tag == "x3ird":
            aux_inputs = np.ndarray([0, 1, 1], dtype=float)
            self.trainer.train(aux_inputs)
            plot_x3(self.sess, self.model, self.test_loader)
        else:
            self.trainer.train(aux_inputs)

    def test(self, X, n_samples=5):
        feed_dict = {
                self.model.inputs: X,
                self.model.is_training: False,
                self.model.n_particles: n_samples,
        }
        return self.sess.run(self.model.outputs, feed_dict=feed_dict)


def plot_x3(sess, model, test_loader, particles=4):
    # Only applicable to config-loaded X3 dataset on IRD feat network.
    X = []
    Y_hat = []
    Y = []

    for x, y in iter(test_loader):
        x = np.expand_dims(x, axis=-1)
        feed_dict = {
                model.main_inputs: x,
                model.is_training: False,
                model.n_particles: particles,
        }
        y_hat = sess.run(model.main_outputs, feed_dict=feed_dict)

        X.extend(x.flatten())
        Y_hat.extend(np.array(y_hat).flatten())
        Y.extend(np.array(y).flatten())

        # Break because the next sess.run will draw new particles.
        break

    N = len(Y)
    for i in range(particles):
        plt.scatter(X, Y_hat[N*i:N*(i+1)], label="prediction {}".format(i),
                s=1)
    plt.scatter(X, Y, label="test", s=3, marker="*")
    plt.legend()
    plt.show()
