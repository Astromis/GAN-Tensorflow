import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class BaseClass:
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.is_trained = False
        self.sess = None
        self.loss = None
        self.saver = None
        self.optimizer = None
        self.logits = None

    def ret_logits(self):
        return self.logits

    def train(self, inp, tar):
        self.sess.run(self.optimizer, feed_dict={self.inputs: inp, self.targets: tar})

    def predict(self, data):
        predicted = self.sess.run(self.logits, feed_dict={self.inputs: data})
        return predicted

    def save_model(self, exp_dir, inp, out):
        tf.saved_model.simple_save(self.sess, exp_dir, inp, out)

    def restore_model(self, exp_dir):
        tf.saved_model.loader.load(self.sess, [tag_constants.TRAINING], exp_dir)

    def check_point(self, file_path):
        self.saver.save(self.sess, file_path)

    def load_checkpoint(self, file_path):
        self.saver.restore(self.sess, file_path)
