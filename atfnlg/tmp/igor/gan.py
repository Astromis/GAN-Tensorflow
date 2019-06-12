import tensorflow as tf
from atfnlg.tmp.igor.generator import Generator
from atfnlg.tmp.igor.discriminator import Discriminator
from atfnlg.tmp.igor.CharNN_generator import CharNN
from .utils import rand_gen, batch_gen, data_preparing, text2tensor, create_gradients_board


class Gan:
    def __init__(self, gan_params, gen_params, discr_params, sess, use_conv):
        self.params = gan_params
        self.sess = sess
        self.real_data = tf.placeholder(tf.int32, (None, self.params['seq_len']), 'real_input')
        self.real_data_ohe = tf.one_hot(self.real_data, self.params['input_vocab'], 1.0, 0.0, axis=2, dtype=tf.float32)
        if use_conv:
            self.real_data_ohe = self.apply_conv(self.real_data_ohe)
        self.var_gen_list = None
        self.var_disc_list = None

        with tf.variable_scope("generator"):
            if self.params['generator_type'] == "charNN":
                self.generator = CharNN(self.sess, gen_params, targets=False)
            elif self.params["generator_type"] == "seq2seq":
                self.generator = self.generator = Generator(self.sess, gen_params)
            else:
                raise NotImplementedError("Generator '{}' not implemented.".format(self.params["generator_type"]))

        self.g_samples = self.generator.ret_logits()
        self.var_gen_list = tf.trainable_variables(scope="generator")

        with tf.variable_scope("discriminator"):
            discriminator = Discriminator(sess, discr_params)

            with tf.variable_scope("real_data"):
                self.d_real = discriminator.run(self.real_data_ohe)

            with tf.variable_scope("fake_data", reuse=tf.AUTO_REUSE):
                self.d_fake = discriminator.run(self.g_samples)

        self.var_disc_list = tf.trainable_variables(scope="discriminator")

        # loss. Note: targets construct as block of ones answers and then follow zeros answers, that actually incorrect
        # Loss for discriminator
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real,
                                                                                  labels=tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,
                                                                                  labels=tf.zeros_like(self.d_fake)))
        self.d_loss = self.d_loss_fake + self.d_loss_real

        # Loss for generator
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,
                                                                             labels=tf.ones_like(self.d_fake)))

        # optimizers
        self.d_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.d_loss,
                                                                            var_list=self.var_disc_list, name="discr_opt")
        self.g_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.g_loss,
                                                                            var_list=self.var_gen_list, name="gener_opt")

    def apply_conv(self, input):
        # fn = lambda x: tf.nn.convolution(x, tf.random_uniform([3,49,49]), "SAME")
        # return tf.fn_map(fn, input)
        conv_inp = tf.nn.conv1d(input, tf.random_uniform([5, 49, 49]), 1, "SAME")
        return conv_inp


    def train(self):
        seq_data, lookup_table = data_preparing(self.params)

        gradients_gen = tf.train.AdamOptimizer(learning_rate=2e-4).compute_gradients(self.g_loss,
                                                                                     var_list=self.var_gen_list)
        gradients_discr = tf.train.AdamOptimizer(learning_rate=2e-4).compute_gradients(self.g_loss,
                                                                                       var_list=self.var_disc_list)
        # Tensorboard
        tf.summary.scalar("Discriminator&Generator loss", self.d_loss)
        tf.summary.scalar("Generator loss", self.g_loss)
        create_gradients_board(gradients_gen)
        create_gradients_board(gradients_discr)
        # Due to in customLSTM calculated softmax of previous state, batch size restricted by size of weights static matrix _y_w
        # But in others type cells this restriction is not.
        # tf.summary.text("Test text,", text2tensor(self.generator.sampling(32, 28, lookup_table)))
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.params["log_dir"] + "test5", graph=self.sess.graph)

        print("Start training...")
        step = 0
        for i in range(self.params['epoch']):
            dis_train_count = 1
            while dis_train_count % self.params['k'] != 0:
                for j, x in enumerate(batch_gen(self.params["batch_size"], seq_data)):

                    tmp, dloss, tb_log_dis = self.sess.run([self.d_solver, self.d_loss, merged_summary_op],
                                                           feed_dict={self.generator.inputs:
                                                                      rand_gen(self.params["batch_size"],
                                                                               self.params['seq_len']),
                                                                      self.real_data: x})  # seq_data[:128, :]
                    summary_writer.add_summary(tb_log_dis, step)
                    step += 1
                    if j == 50:
                        break
                dis_train_count += 1
            for _ in range(10):
                for j, x in enumerate(batch_gen(self.params["batch_size"], seq_data)):
                    tmp, gloss, tb_log_gen = self.sess.run([self.g_solver, self.g_loss, merged_summary_op],
                                                           feed_dict={self.generator.inputs:
                                                                          rand_gen(self.params["batch_size"],
                                                                           self.params['seq_len']), self.real_data: x})
                    summary_writer.add_summary(tb_log_gen, step)
                    step += 1
                    # tf.summary.text("Test text,", text2tensor(self.generator.sampling(32, 28, lookup_table)))
                    if j == 50:
                        break

            print("Generator loss: %f ; Discriminator loss: %f" % (gloss, dloss))

    # def train_on_batch(self):
    #    train_params = tf.trainable_variables()
    #    gradients = tf.gradients(self.loss, train_params)
    #    # How to get a max gradient norm? And what is this?
    #    self.update = self.optimizer.applt_gradients(zip(gradients, train_params))
