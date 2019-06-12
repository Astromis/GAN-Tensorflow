import tensorflow as tf
from tensorflow.contrib import rnn
from atfnlg.tmp.igor.BaseClassfile import BaseClass
import numpy as np
from atfnlg.tmp.igor.custom_lstm import CustomLSTMCell


class CharNN(BaseClass):
    def __init__(self, sess, hparam, targets=True, one_hot=True):
        super().__init__()
        self.inputs = tf.placeholder(tf.int32, (None, hparam["seq_len"]), 'inputs')  # hparam["num_input"]
        if targets:
            self.targets = tf.placeholder(tf.float32, (None, hparam["num_classes"]), 'outputs')
        if one_hot:
            self.inputs_ohe = tf.one_hot(self.inputs, hparam['input_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32)
            if targets:
                self.targets_ohe = tf.one_hot(self.outputs, hparam['output_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32)
        self.sess = sess
        self.cell = None
        self.stacked_cells = None
        self.cells = []
        W = tf.get_variable("weights_dense1", shape=(hparam["num_hidden"], hparam["num_classes"]),
                            initializer=tf.random_normal_initializer)
        self.RNN = self.stacking_cells(hparam["cell_type"], hparam["num_hidden"], hparam["num_layers"],
                                       hparam["dropout"])

        # Get lstm cell output
        self.outputs, self.states = tf.nn.dynamic_rnn(self.RNN, self.inputs_ohe, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        self.dense_1 = tf.contrib.layers.fully_connected(self.outputs, num_outputs=hparam["num_classes"],
                                                         activation_fn=None)
        # print("Output tensor: ", tf.reduce_mean(self.outputs, axis=1))
        # print("Output tensor: ", self.dense_1)
        # self.dense_1 = tf.matmul(tf.reshape(self.outputs,
        #[self.outputs[-1].get_shape()[0] * self.outputs[-1].get_shape()[1], 49]), W, transpose_b=True)
#        self.dense_1 = tf.matmul(tf.reduce_mean(self.states, axis=0), W)
        self.logits = tf.nn.softmax(self.dense_1)
        # probably don't need define loss and optimizer inside the class
        # self.set_loss(hparam["loss"])
        # self.set_optimizer(hparam["opt"], hparam["learning_rate"])
        # return tf.matmul(outputs[-1], weights['out']) + biases['out']
        self.saver = tf.train.Saver()

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
        # setting up an optimizer
        if opt == "grad_desc":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        elif opt == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        elif opt == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)
        # elif ...

    def set_loss(self, loss):
        # setting up a loss function
        if loss == 'cross_entropy':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=self.targets))
        # elif loss == ...

    def train(self, inp, tar):
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.inputs: inp, self.targets: tar})
        self.is_trained = True
        return _, loss

    def predict_distribution(self, data):
        predicted = self.sess.run(self.logits, feed_dict={self.inputs: data})
        return predicted

    def rand_gen(self, batch_size, seq_len):
        return np.random.randint(0, high=seq_len, size=(batch_size, seq_len))

    def sampling(self, batch_size, seq_len, num2char):
        data = self.rand_gen(batch_size, seq_len)
        pred = self.predict_distribution(data)
        pred_sampled = pred.argmax(axis=2)
        test = []
        for word in pred_sampled:
            string = []
            for al in word:
                string.append(num2char[al])
            test.append("".join(string))
        #return "".join(string)
        print(test)
        return test
        pass

