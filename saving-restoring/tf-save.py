"""Saving & Loading demo in TensorFlow."""

import os

# Suppress tensorflow deprecation warning
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

# Save directory.
save_dir = os.path.join('saved/saving-restoring', 'tf-save')

# Create the save directory if it doesn't exist.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Definition of two variables in the default graph.
v1 = tf.get_variable("v1", shape=[2, 3], initializer=tf.ones_initializer)
v2 = tf.get_variable("v2", shape=[3, 2], initializer=tf.random_normal_initializer)

# Perform an add operation.
mul = tf.matmul(v1, v2, name='mul')

# Check if the add operation is in the default graph.
print('mul in default graph? {}'.format(mul.graph == tf.get_default_graph()))

# Initialize all global variables operation.
init_op = tf.global_variables_initializer()

# Saver objects.
# By default tf.train.Saver handles all the variables in the default graph.
all_saver = tf.train.Saver()

# Saver for v2
# You can also specify which variables to save (as a list) or under
# what name (as a dictionary).
v2_saver = tf.train.Saver({'v2': v2})

# NOTE: You must be careful to use a Saver with a Session linked to
# the Graph containing all the variables the Saver is handling.
# By default the Session handles the default graph & all related variables.
with tf.Session() as sess:
    # Initialize all global variables.
    # sess.run(init_op)
    
    # Initialize only v1, since we're going to be restoring v2.
    v1.initializer.run()

    # We save variables after creating a Session object.
    # all_saver.save(sess, os.path.join(save_dir, 'all-vars.ckpt'))

    # Or you can save a handful of variables.
    # NOTE: v2 is in the default graph, that's why we can pass the
    # current Session because it handles the default graph as well.`
    v2_saver.restore(sess, os.path.join(save_dir, 'v2-var.ckpt'))

    print('v1.eval() = {}'.format(v1.eval()))
    print('v2.eval() = {}'.format(v2.eval()))
    print('mul.eval() = {}'.format(mul.eval()))
