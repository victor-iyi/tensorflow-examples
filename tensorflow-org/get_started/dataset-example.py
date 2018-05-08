"""Proper way to feed data into TensorFlow models.

  @author
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola

  @project
    File: dataset-example.py
    Created on 08 May, 2018 @ 04:46 AM.

  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import numpy as np

import tensorflow as tf

# METHOD 1: From single numpy array.
x = np.random.sample(size=(100, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# METHOD 2: Passing more than one array.
x = np.random.sample((100, 2))
y = np.random.sample((100, 1))

dataset = tf.data.Dataset.from_tensor_slices((x, y))

# METHOD 3: From Tensors.
x = tf.random_normal(shape=(100, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# METHOD 4: From Placeholders:
# (in case we want to dynamically change data inside the Dataset)
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
dataset = tf.data.Dataset.from_tensor_slices(x)

# METHOD 5: From Generators.
# (useful when we have an array of different elements length e.g. sequence)
sequence = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.int64)


def generator():
    """Generator function."""
    for seq in sequence:
        yield seq


# Create Dataset object from generator function
dataset = tf.data.Dataset.from_generator(generator,
                                         output_types=tf.int64,
                                         output_shapes=(None, 2))

# Create an iterator for enumerating elements in the dataset.
iterator = dataset.make_initializable_iterator()

# get next item in the `iterator`
seq = iterator.get_next()

with tf.Session() as sess:
    # `dataset.make_initializable_iterator` returns uninitialized iterator.
    # therefore, we need to initialize it before using it.
    sess.run(iterator.initializer)
    print('seq = {}'.format(sess.run(seq)))
