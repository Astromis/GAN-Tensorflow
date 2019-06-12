import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from atfnlg.tmp.igor.BaseClassfile import BaseClass


class Generator(BaseClass):
    def __init__(self, sess, hparams, targets=True):
        super().__init__()
        self.sess = sess
        # placeholders
        self.inputs = tf.placeholder(tf.int32, (None, hparams['x_seq_len']), 'inputs')
        # self.outputs = tf.placeholder(tf.int32, (None, hparams['y_seq_len']), 'output')
        if targets:
            self.targets = tf.placeholder(tf.int32, (None, None), 'targets')
        # one-hot encoding
        input_ohe = tf.one_hot(self.inputs, hparams['input_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32, name="ohe_input")
        # output_ohe = tf.one_hot(self.outputs, hparams['output_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32, name="ohe_output")

        # W = tf.get_variable("weights_dense_gen", shape=(-1, hparams['o_voc']), initializer=tf.random_normal_initializer)

        # encoder
        with tf.variable_scope("encoder") as encoder_scope:
            encoder = self.set_cell_type(hparams['cell_type'], hparams['num_hidden'])
            enc_outputs, last_state = tf.nn.dynamic_rnn(encoder, inputs=input_ohe, dtype=tf.float32)
        # decoder
        with tf.variable_scope("decoder") as decoder_scope:
            decoder = self.set_cell_type(hparams['cell_type'], hparams['num_hidden'])
            dec_outputs, _ = tf.nn.dynamic_rnn(decoder, inputs=enc_outputs, initial_state=last_state)

        self.dense_1 = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=hparams['output_vocab'], activation_fn=None)
        #self.dense_1 = tf.matmul(dec_outputs, W,)
        self.logits = tf.nn.softmax(self.dense_1, name="gen_softmax")

        self.saver = tf.train.Saver()

    def set_cell_type(self, c_type, n_hid):
        # choosing rnn type
        if c_type == "gru":
            return rnn.BasicLSTMCell(n_hid)
        elif c_type == "lstm":
            return rnn.GRUCell(n_hid)
        elif c_type == "rnn":
            return rnn.BasicRNNCell(n_hid)

    def set_optimizer(self, opt, learning_rate):
        # setting up an optimizer
        if opt == "grad_desc":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, name="gen_rnn_opt")
        elif opt == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss, name="gen_rnn_opt")
        elif opt == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss, name="gen_rnn_opt")

    def set_loss(self, batch_size, y_seq_len):
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, tf.ones([batch_size,
                                                                                         y_seq_len]))

    def predict_distribution(self, data):
        return self.sess.run(self.logits, feed_dict={self.inputs: data})

    def sampling(self, data_input, seq_len, num2char_y, char2num_y, file_path):
        dec_input = np.zeros((len(data_input), 1)) + char2num_y['<GO>']
        for i in range(seq_len):
            batch_logits = self.sess.run(self.logits,
                                         feed_dict={self.inputs: data_input,
                                                    self.outputs: dec_input})
            prediction = batch_logits[:, -1].argmax(axis=-1)
            dec_input = np.hstack([dec_input, prediction[:, None]])

        dest_chars = [num2char_y[l] for l in dec_input[np.random.randint(0, len(data_input)), 1:]]
        dest_chars = "".join(dest_chars)
        f = open(file_path, 'w')
        f.write(dest_chars)
        f.close()
        return dest_chars

    def train(self, source_batch, target_batch):
        _, batch_loss, batch_logits = self.sess.run([self.optimizer, self.loss, self.logits],
                                                    feed_dict={self.inputs: source_batch,
                                                               self.outputs: target_batch[:, :-1],
                                                               self.targets: target_batch[:, 1:]})
        self.is_trained = True
        return batch_loss, batch_logits


