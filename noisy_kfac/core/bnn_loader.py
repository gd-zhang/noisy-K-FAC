from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ..misc.utils import get_logger, get_args, makedirs
from ..misc.config import process_config
from ..misc.data_loader import load_pytorch
from .model import Model
from .train import Trainer

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
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(parent_dir, '../../')
        configpath = os.path.join(root_dir, 'configs/', config_name)
        self.config = config = process_config(configpath)

        # set logger
        path1 = os.path.join(root_dir, 'noisy_kfac/core/model.py')
        path2 = os.path.join(root_dir, 'noisy_kfac/core/train.py')
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

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with self.sess.as_default():
                # define computational graph
                input_size = input_size or _INPUT_DIM[self.config.dataset]
                self.model = Model(self.config, input_size,
                        len(self.train_loader))
                self.trainer = Trainer(self.sess, self.model, self.train_loader,
                        self.test_loader, self.config, self.logger)

        self.train = self.use_graph_and_sess(self._train)
        self.test = self.use_graph_and_sess(self._test)

    def use_graph_and_sess(self, f):
        """
        Decorate a function to use the graph and session associated with this
        BNNLoader.
        """
        def inner(*args, **kwargs):
            with self.graph.as_default():
                with self.sess.as_default():
                    return f(*args, **kwargs)
        return inner


    def _train(self, aux_inputs):
        self.trainer.train(aux_inputs)

    def _test(self, X, n_samples=5):
        feed_dict = {
                self.model.inputs: X,
                self.model.is_training: False,
                self.model.n_particles: n_samples,
        }
        return self.sess.run(self.model.outputs, feed_dict=feed_dict)


def plot_x3(sess, model_, test_loader, particles=5):
    # Only applicable to config-loaded X3 dataset.
    X, Y = test_loader.dataset.tensors
    feed_dict = {
            model_.inputs: X,
            model_.is_training: False,
            model_.n_particles: particles,
    }
    Y_hat = sess.run(model_.outputs, feed_dict=feed_dict).flatten()
    X = X.numpy().flatten()
    Y = Y.numpy()

    N = len(Y)
    for i in range(particles):
        plt.scatter(X, Y_hat[N*i:N*(i+1)], label="prediction {}".format(i),
                s=1)
    plt.scatter(X, Y, label="test", s=3)
    plt.legend()
    plt.show()
