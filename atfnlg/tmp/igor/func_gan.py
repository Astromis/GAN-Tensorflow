import tensorflow as tf
from tensorflow.contrib import rnn


# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

x_ohe = tf.one_hot(X, hparams['input_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32, name="real_data_ohe")
z_ohe = tf.one_hot(Z, hparams['input_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32, name="noise_data_ohe")

D_W1 = tf.get_variable('D_W1', shape=(), initializer=tf.contrib.layers.xavier_initializer)
D_b1 = tf.get_variable("D_b1", shape=(), initializer=tf.zeros_initializer)

D_W2 = tf.get_variable('D_W2', shape=(), initializer=tf.contrib.layers.xavier_initializer)
D_b2 = tf.get_variable("D_b2", shape=(), initializer=tf.zeros_initializer)


theta_D = [D_W1, D_W2, D_b1, D_b2]

G_W1 = tf.get_variable('G_W1', shape=(), initializer=tf.contrib.layers.xavier_initializer)
G_b1 = tf.get_variable("G_b1", shape=(), initializer=tf.zeros_initializer)

G_W2 = tf.get_variable('G_W2', shape=(), initializer=tf.contrib.layers.xavier_initializer)
G_b2 = tf.get_variable("G_b2", shape=(), initializer=tf.zeros_initializer)

theta_G = [G_W1, G_W2, G_b1, G_b2]


def stacking_cells(c_type, n_hid, n_lay, dropout=0.5):
    cells =[]
    # choosing rnn type
    if c_type == "gru":
        cell = rnn.BasicLSTMCell(n_hid, forget_bias=1.0)
    elif c_type == "lstm":
        cell = rnn.GRUCell(n_hid)
    elif c_type == "rnn":
        cell = rnn.BasicRNNCell(n_hid)
    # wrapping in dropout
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
    # stacking
    for _ in range(n_lay):
        cells.append(cell)
    stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
    return stacked_cells


def generator(z, cell_type, num_hiden, num_layers):
    RNN = stacking_cells(cell_type, num_hiden, num_layers)
    outputs, states = tf.nn.dynamic_rnn(RNN, z, dtype=tf.float32)

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x, cell_type, num_hidden, num_layers, dropout):
    RNN = stacking_cells(cell_type, num_hidden,
                         num_layers, dropout)
    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(RNN, x, dtype=tf.float32)

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(z_ohe, 'lstm', 64, 1)
D_real, D_logit_real = discriminator(x_ohe, 'lstm', 64, 1)
D_fake, D_logit_fake = discriminator(G_sample, 'lstm', 64, 1)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

#  Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})