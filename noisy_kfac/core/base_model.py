import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()

    @property
    def trainable_variables(self):
        # note: we don't train the params of BN
        vars = []
        for var in tf.trainable_variables():
            if "train_w_" in var.name:
                vars.append(var)
        return vars

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        self.global_step_tensor = tf.train.get_or_create_global_step()

    def init_optim(self):
        raise NotImplementedError

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError
