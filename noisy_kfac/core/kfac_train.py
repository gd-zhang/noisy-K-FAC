from .base_train import BaseTrain
from tqdm import tqdm
import numpy as np

from . import MODE_IRD


class KFACTrainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(KFACTrainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, aux_inputs=None):
        if self.model.mode == MODE_IRD:
            base_feed_dict = {
                self.model.aux_inputs: aux_inputs,
            }
        else:
            base_feed_dict = {}
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch(base_feed_dict)
            self.test_epoch(base_feed_dict)

    def train_epoch(self, base_feed_dict={}):
        loss_list = []
        acc_list = []
        for itr, data in enumerate(tqdm(self.train_loader)):
            feed_dict = dict(base_feed_dict)
            if self.model.mode == MODE_IRD:
                x = data
            else:
                assert self.model.targets is not None
                feed_dict[self.model.targets] = y
                x, y = data

            feed_dict.update({
                self.model.is_training: True,
                self.model.n_particles: self.config.train_particles,
                self.model.inputs: x,
                })

            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # note: that's important

            if self.model.mode == MODE_IRD:
                loss, outputs, aux_outputs, total_loss = self.sess.run([
                    self.model.loss, self.model.outputs,
                    self.model.aux_outputs, self.model.total_loss],
                    feed_dict=feed_dict)
            else:
                loss = self.sess.run([self.model.loss], feed_dict=feed_dict)
            loss_list.append(loss)

            cur_iter = self.model.global_step_tensor.eval(self.sess)
            if cur_iter % self.config.TCov == 0:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.TInv == 0:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)

        avg_loss = np.mean(loss_list)

        print("train | loss: %5.4f "%(float(avg_loss)))

        # summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        if self.model.mode == MODE_IRD:
            print("train | total_loss: %5.4f "%(float(total_loss)))
            print("train | aux_output: %5.4f "%(float(aux_outputs[0])))
            print("train | output: %5.4f "%(float(outputs[0])))
            # summaries_dict['Z_hat'] = aux_outputs[0]
            # summaries_dict['predicted_reward'] = outputs[0]

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        # self.model.save(self.sess)

    def test_epoch(self, base_feed_dict={}):
        loss_list = []
        acc_list = []
        for data in self.test_loader:
            feed_dict = dict(base_feed_dict)
            if self.model.mode == MODE_IRD:
                x = data
            else:
                x, y = data
                assert self.model.targets is not None
                feed_dict[self.model.targets] = y
            feed_dict.update({
                self.model.inputs: x,
                self.model.is_training: False,
                self.model.n_particles: self.config.test_particles,
                })

            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("test | loss: %5.4f | accuracy: %5.4f\n"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)
