import tensorflow as tf
from atfnlg.tmp.igor.BaseClassfile import BaseClass
from atfnlg.tmp.igor.custom_lstm import CustomLSTMCell
from tensorflow.contrib import rnn


class Discriminator(BaseClass):
    def __init__(self, sess, hparams, targets=True):
        super().__init__()
        if targets:
            self.targets = tf.placeholder(tf.int32, (None, 2), 'targets')
        self.sess = sess
        self.saver = tf.train.Saver()
        self.hparams = hparams

    def run(self, ch_input):
        inputs = ch_input
        W = tf.get_variable("weights_dense2", shape=(40, 2), initializer=tf.random_normal_initializer)#tf.Variable(tf.random_normal(shape=()))
        RNN = self.stacking_cells(self.hparams['cell_type'], self.hparams['num_hidden'],
                                  self.hparams['num_layers'], self.hparams['dropout'])
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(RNN, inputs, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        dense_1 = tf.contrib.layers.fully_connected(outputs[:, -1, :], num_outputs=40, activation_fn=None)
        # dense_2 = tf.contrib.layers.fully_connected(dense_1, num_outputs=2)
        dense_2 = tf.matmul(dense_1, W)
        logits = tf.nn.sigmoid(dense_2, name="discr_softmax")

        return logits

    def stacking_cells(self, c_type, n_hid, n_lay, dropout):
        def call_cell(cell_type, nun_hid, dropout):
            cell = None
            # choosing rnn type
            if cell_type == "gru":
                cell = rnn.GRUCell(nun_hid)
            elif cell_type == "lstm":
                cell = CustomLSTMCell(nun_hid, forget_bias=1.0)
            elif cell_type == "rnn":
                cell = rnn.BasicRNNCell(nun_hid)
            # wrapping in dropout
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)

        # stacking
        self.cells = tf.contrib.rnn.MultiRNNCell([call_cell(c_type, n_hid, dropout) for _ in range(n_lay)], state_is_tuple=True)
        return self.cells

    def set_optimizer(self, opt, learning_rate):
        if self.loss is None:
            print("Loss is not defined. Use set_loss method")
            raise ValueError
        # setting up an optimizer
        if opt == "grad_desc":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, name="discr_rnn_opt")
        elif opt == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss, name="discr_rnn_opt")
        elif opt == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss, name="discr_rnn_opt")
        else:
            raise NotImplementedError("Optimizer '{}' not implemented.".format(opt))

        return self

    def set_loss(self, loss):
        if self.targets is None:
            print("Target placeholder is not defined.")
            raise ValueError
        # setting up a loss function
        if loss == 'cross_entropy':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=self.targets))
        else:
            raise NotImplementedError("Optimizer '{}' not implemented.".format(loss))

        return self

    def train(self, inp, tar):
        self.sess.run(self.optimizer, feed_dict={self.inputs: inp, self.targets: tar})
        self.is_trained = True

    def predict(self, data):
        predicted = self.sess.run(self.logits, feed_dict={self.inputs: data})
        return predicted
