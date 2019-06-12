from .CharNN_generator import CharNN
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
tf.reset_default_graph()


# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

sess = tf.InteractiveSession()
model = CharNN(sess, timesteps, num_input, num_classes, 0.5)

# Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(model.ret_logits(), 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

for step in range(1, training_steps+1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, timesteps, num_input))
    # Run optimization op (backprop)
    _, loss = model.train(batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        # loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
        #                                                     Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

print("Optimization Finished!")

# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
test_label = mnist.test.labels[:test_len]
# print("Testing Accuracy:",\
# sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
