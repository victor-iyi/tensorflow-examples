"""Saving & Loading demo in TensorFlow."""

# Suppress tensorflow deprecation warning
import warnings
warnings.filterwarnings('ignore')

import os
import tensorflow as tf

save_dir = os.path.join('saved/tf-save')

# Create the save directory if it doesn't exist.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Definition of two variables in the default graph.
v1 = tf.Variable(1, name='v1')
v2 = tf.Variable(2, name='v2')

# Perform an add operation.
add = tf.add(v1, v2, name='add')

# Check if the add operation is in the default graph.
print('add in default graph? {}'.format(add.graph == tf.get_default_graph()))

# Saver objects.
# By default tf.train.Saver handles all the variables in the default graph.
all_saver = tf.train.Saver()

# Saver for v2
# You can also specify which variables to save (as a list) or under
# what name (as a dictionary).
v2_saver = tf.train.Saver({'v2': v2})

# By default the Session handles the default graph & all related variables.
with tf.Session() as sess:
    # Initialize all global variables.
    init = tf.global_variables_initializer()
    sess.run(init)

    # We save variables after creating a Session object.
    all_saver.save(sess, os.path.join(save_dir, 'all-vars.ckpt'))

    # Or you can save a handful of variables.
    # NOTE: v2 is in the default graph, that's why we can pass the
    # current Session because it handles the default graph as weell.`
    v2_saver.save(sess, os.path.join(save_dir, 'v2-var.ckpt'))
