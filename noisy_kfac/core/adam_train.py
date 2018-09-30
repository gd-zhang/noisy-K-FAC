from .base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class NoisyAdamTrainer(BaseTrain):

    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(NoisyAdamTrainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader


    def train(self, aux_inputs=None):
        # XXX: Later when testing x3 distribution for Noisy Adam,
        # Do this thing where I move base_feed_dict into the model
        # instead so that I can set aux_inputs or other shit outside of
        # train. In other words, the model gets to choose whether or not
        # there is an aux_input inside the base_feed_dict.
        #
        # In other words, the train() that ird-plus will be calling should
        # belong to a higher wrapper. Not this train class.
        base_feed_dict = {
            self.model.aux_inputs: aux_inputs,
        }

        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch(base_feed_dict)
            self.test_epoch(base_feed_dict)


    def train_epoch(self, base_feed_dict={}):
        fd = dict(base_feed_dict)
        fd[self.model.n_particles] = self.config.train_particles
        _epoch(base_feed_dict=fd, is_training=True)


    def test_epoch(self, base_feed_dict={}):
        fd = dict(base_feed_dict)
        fd[self.model.n_particles] = self.config.test_particles
        _epoch(base_feed_dict=fd, is_training=False)


    def _epoch(self, is_training, base_feed_dict={}):
        """
        Base function for train_epoch and test_epoch.
        Params:
        is_training: bool. Whether or not we are training right now.
        base_feed_dict: dict. Prepopulated feed_dict. Add model-specific
          feed_dict entries here. This function will deal automatically
          with inputs and targets.
        """

        if None in base_feed_dict:
            raise ValueError("Invalid key 'None' in base_feed_dict",
                    base_feed_dict)

        loss_list = []
        for itr, data in enumerate(tqdm(self.train_loader)):
            feed_dict = dict(base_feed_dict)

            # if self.model.mode == MODE_IRD:
            if len(data) == 1:
                # e.g.:, for IRD with no targets
                x = data
            elif len(data) == 2:
                # e.g., for x3 data with input and targets
                x, y = data
                feed_dict[self.model.targets] = y
            else:
                raise ValueError("don't know how to interpret data".format(
                    feed_dict))

            feed_dict.update({
                self.model.is_training: is_training,
                self.model.n_particles: self.config.train_particles,
                self.model.inputs: x,
                })

            # XXX: Important!
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            # idk why this is important, but ok. Oh, maybe batch norm and stuff?
            feed_dict[self.model.is_training] = False  # note: that's important

            # TODO: Implement this method. loss is obvious.
            # log_aux is a dictionary -- {str => scalar}. It contains other
            # quantities to log.
            loss, log_aux = self.model.get_loss_and_aux(feed_dict=feed_dict)
            if self.model.mode == MODE_IRD:
                loss, outputs, aux_outputs, total_loss = self.sess.run([
                    self.model.loss, self.model.outputs,
                    self.model.aux_outputs, self.model.total_loss],
                    feed_dict=feed_dict)
            else:
                loss = self.sess.run([self.model.loss], feed_dict=feed_dict)
            loss_list.append(loss)

            cur_iter = self.model.global_step_tensor.eval(self.sess)

            # XXX: Make sure to update mu and f here. Figure out how.
            # ???

        avg_loss = np.mean(loss_list)

        prefix = "train" if is_training else "test"
        print("{} | loss: {:5.4f} ".format(prefix, float(avg_loss)))

        # summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss

        for name, scalar in aux.items():
            print("{} | {}: {:5.4f} ".format(prefix, name, scalar)

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        # self.model.save(self.sess)
