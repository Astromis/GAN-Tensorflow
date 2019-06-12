import tensorflow as tf
from atfnlg.tmp.igor.gan import Gan

# I know train functions seems not so like we discussed
# In importing with dot on the left side of filename, for example .gan, pycharm get an error and I can't figure out why.
# So I temporary delete it.
# Check out please gan code in 26 line.

# Loss don't decrease.

chnn_param = {"seq_len": 28,
              "num_input": 28,
              "num_classes": 49,
              "cell_type": "lstm",
              "num_hidden": 32,
              "num_layers": 1,
              "dropout": 0.5,
              "loss": "cross_entropy",
              "opt": "rmsprop",
              "learning_rate": 0.1,
              "input_vocab": 58
              }

seq2seq_param = {"x_seq_len": 28,
                 "y_seq_len": 10,
                 "input_vocab": 49,
                 "output_vocab": 49,
                 "cell_type": "gru",
                 "num_hidden": 64
                }
discr_params = {"dropout": 0.5,
                "cell_type": "gru",
                'num_layers': 1,
                "num_hidden": 64,
                'opt': "rmsprop",
                'learning_rate': 0.1,
                'loss': "cross_entropy",
                "inp_seq_len": 10,
                "i_voc": 13,
                "o_voc": 10
                }

gan_params = {'seq_len': 28,
              'input_vocab': 49,
              "generator_type": "seq2seq",
              "file": '/home/astromis/PycharmProjects/ATFNLG/data/english/nitshe.txt',
              "batch_size": 32,
              "k": 2,
              "epoch": 10,
              "log_dir": "./tboard_logs/"
              }

sess = tf.InteractiveSession()

model = Gan(gan_params, seq2seq_param, discr_params, sess, True)
sess.run(tf.global_variables_initializer())

model.train()
