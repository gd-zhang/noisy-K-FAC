import tensorflow as tf
from misc.utils import Summarizer


class BaseTrain:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.summarizer = Summarizer(sess, config)
        self.config = config
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def test_epoch(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
