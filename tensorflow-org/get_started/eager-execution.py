"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: eager-execution.py
    Created on 03 May, 2018 @ 6:17 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from .iris_data import maybe_download, csv_input_fn

# Turn on eager execution.
tf.enable_eager_execution()
print("Eager execution status: {}".format(tf.executing_eagerly()))

# Download and load the dataset.
train_path, test_path = maybe_download()
train_data = csv_input_fn(train_path, batch_size=32)
test_data = csv_input_fn(test_path)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(units=10, activation="relu"),
    tf.keras.layers.Dense(units=10)
])


def loss(model, x, y):
    y_hat = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)

