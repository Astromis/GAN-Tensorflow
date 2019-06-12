import tensorflow as tf
import numpy as np

from atfnlg.tmp.igor.discriminator import Discriminator


sess = tf.InteractiveSession()
params = {"dropout": 0.5, "cell_type": "gru", 'num_layers': 1, "num_hidden": 32,
          'opt': "grad_desc", 'learning_rate': 0.1, 'loss': "cross_entropy", "inp_seq_len": 10,
          "i_voc": 13, "o_voc": 10}

ph1 = tf.placeholder(tf.float32, (None, 20, 10), 'inputs')
ph2 = tf.placeholder(tf.float32, (None, 30, 10), 'inputs')
with tf.variable_scope("disc") as scope:
    model = Discriminator(sess, ph1, params)
with tf.variable_scope("disc", reuse=True) as scope:
    model = Discriminator(sess, ph2, params)

data = np.random.normal(size=(128, 10))
target = np.random.normal(size=(128, 2))
sess.run(tf.global_variables_initializer())

model.train(data, target)
