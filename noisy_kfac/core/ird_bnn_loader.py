from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from ..misc.utils import get_logger, get_args, makedirs
from ..misc.config import process_config
from ..misc.data_loader import load_pytorch
from .model import Model
from .train import Trainer

class IRDBNNLoader():
    logger = None

    def __init__(self, input_size=None, config_name="ird/kfac.json"):
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

        # Typically this is where we load dataset. However, IRD has a
        # dynamically generated dataset instead. Confirm that our config
        # doesn't in fact specify a dataset.
        assert data not in config

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
