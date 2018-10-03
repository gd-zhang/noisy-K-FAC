from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from noisy_kfac.misc.utils import get_logger, get_args, makedirs
from noisy_kfac.misc.config import process_config
from noisy_kfac.misc.data_loader import load_pytorch
from noisy_kfac.core.kfac_model import KFACModel
from noisy_kfac.core.kfac_train import KFACTrainer
from noisy_kfac.core.bnn_loader import BNNLoader


_INPUT_DIM = {
    'fmnist': [784],
    'mnist': [784],
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3],
    'x3': [1],
}


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)
    from_config(config)

def bnn_main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    args = get_args()
    loader = BNNLoader(config_name=args.config)
    loader.train(None)

def from_config(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__))

    logger.info(str(config.items()))

    # load data
    train_loader, test_loader = load_pytorch(config)

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g).as_default() as sess:
            # define computational graph
            model_ = KFACModel(config, _INPUT_DIM[config.dataset], len(train_loader.dataset))
            trainer = KFACTrainer(sess, model_, train_loader, test_loader, config, logger)

            trainer.train()

            if config.dataset == "x3":
                plot_x3(sess, model_, test_loader)

def plot_x3(sess, model_, test_loader, particles=5):
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


if __name__ == "__main__":
    bnn_main()
