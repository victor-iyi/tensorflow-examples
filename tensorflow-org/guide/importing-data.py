"""Proper way to feed data into TensorFlow models.

  @author
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola

  @project
    File: importing-data.py
    Created on 08 May, 2018 @ 04:46 AM.

  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import numpy as np

import tensorflow as tf

######################################################################
# +------------------------------------------------------------------+
# | How to create dataset: (5 methods here...)
# +------------------------------------------------------------------+
######################################################################
# METHOD 1: From single numpy array.
x = np.random.sample(size=(100, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# Clean variables from memory.
del x, dataset

# METHOD 2: Passing more than one array.
x = np.random.sample((100, 2))
y = np.random.sample((100, 1))

dataset = tf.data.Dataset.from_tensor_slices((x, y))

# Clean variables from memory.
del x, y, dataset

# METHOD 3: From Tensors.
x = tf.random_normal(shape=(100, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# Clean variables from memory.
del x, dataset

# METHOD 4: From Placeholders:
# (in case we want to dynamically change data inside the Dataset)
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# Clean variables from memory.
del x, dataset

# METHOD 5: From Generators.
# (useful when we have an array of different elements length e.g. sequence)
print('\nFrom Generators:')
sequence = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.int64)


def generator():
    """Generator function."""
    for seq in sequence:
        yield seq


# Create Dataset object from generator function
dataset = tf.data.Dataset.from_generator(generator,
                                         output_types=tf.int64)

# Create an iterator for enumerating elements in the dataset.
iterator = dataset.make_initializable_iterator()

# get next item in the `iterator`
seq = iterator.get_next()

with tf.Session() as sess:
    # `dataset.make_initializable_iterator` returns uninitialized iterator.
    # therefore, we need to initialize it before using it.
    sess.run(iterator.initializer)
    print('seq = {}'.format(sess.run(seq)))

# Clean variables from memory.
del sequence, generator, dataset, iterator, seq, sess

######################################################################
# +------------------------------------------------------------------+
# | How to create `Iterator` (to retrieve the real values in Dataset).
# +------------------------------------------------------------------+
######################################################################
# METHOD 1: One shot Iterator,
print('\nUsing one shot iterator.')
x = np.random.sample(size=(100, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)
iterator = dataset.make_one_shot_iterator()
elements = iterator.get_next()

with tf.Session() as sess:
    print('elements = {}'.format(sess.run(elements)))
    print('elements = {}'.format(sess.run(elements)))
    print('elements = {}'.format(sess.run(elements)))

# Clean variables from memory.
del x, dataset, iterator, elements, sess

# METHOD 2: Initializable Iterator.
print('\nUsing a placeholder.')
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
dataset = tf.data.Dataset.from_tensor_slices(x)

# Real numpy data (we'll pass it through `feed_dict`).
data = np.random.sample(size=(100, 2))

# Make initializable Iterator.
iterator = dataset.make_initializable_iterator()
elements = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x: data})
    print('elements = {}'.format(sess.run(elements)))
    print('elements = {}'.format(sess.run(elements)))

# Clean variables from memory.
del x, dataset, data, iterator, elements, sess

######################################################################
# +------------------------------------------------------------------+
# | Real world example and initializable dataset.
# +------------------------------------------------------------------+
######################################################################
print('\nReal world example:')

# Training dataset.
X_train = np.random.sample(size=(100, 2))
y_train = np.random.sample(size=(100, 1))

# Testing dataset.
X_test = np.random.sample(size=(10, 2))
y_test = np.random.sample(size=(10, 1))

X_plhd = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_plhd = tf.placeholder(dtype=tf.float32, shape=[None, 1])

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()

epochs = 5
with tf.Session() as sess:
    print('Train data:')
    for epoch in range(epochs):
        # Initialize iterator for training data.
        sess.run(iterator.initializer, feed_dict={X_plhd: X_train,
                                                  y_plhd: y_train})
        _features, _labels = sess.run([features, labels])
        print('features = {}\t labels = {}'.format(_features, _labels))
    print('Test data:')
    # Initialize iterator for testing data.
    sess.run(iterator.initializer, feed_dict={X_plhd: X_test,
                                              y_plhd: y_test})
    _features, _labels = sess.run([features, labels])
    print('features = {}\t labels = {}'.format(_features, _labels))

# Clean variables from memory.
del X_train, y_train, X_test, y_test, dataset, iterator, features, labels
del X_plhd, y_plhd, epochs, sess, _features, _labels

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Re-initializable dataset.
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
print('\nRe-initializable dataset example')

# Training dataset.
X_train = np.random.sample(size=(100, 2))
y_train = np.random.sample(size=(100, 1))

# Testing dataset.
X_test = np.random.sample(size=(10, 2))
y_test = np.random.sample(size=(10, 1))

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

iterator = tf.data.Iterator.from_structure(output_types=train_data.output_types,
                                           output_shapes=train_data.output_shapes)
features, labels = iterator.get_next()

train_init = iterator.make_initializer(dataset=train_data, name="train_dataset")
test_init = iterator.make_initializer(dataset=test_data, name="test_dataset")

with tf.Session() as sess:
    # Train dataset initializer.
    sess.run(train_init)
    X, y = sess.run([features, labels])
    print(X, y)

    # Test dataset initializer.
    sess.run(test_init)
    X, y = sess.run([features, labels])
    print(X, y)

del X_train, X_test, y_train, y_test, train_data, test_data, iterator
del features, labels, train_init, test_init, X, y, sess
