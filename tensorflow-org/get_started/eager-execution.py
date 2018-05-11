"""A demo of eager execution in TensorFlow.

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
from tensorflow.contrib.eager.python import tfe

from iris_data import maybe_download, csv_input_fn

# Turn on eager execution.
tf.enable_eager_execution()
print("Eager execution status: {}".format(tf.executing_eagerly()))

# Download and load the dataset.
train_path, test_path = maybe_download()
train_data = csv_input_fn(train_path, batch_size=32)
test_data = csv_input_fn(test_path)

# Create a model using Keras.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(units=10, activation="relu"),
    tf.keras.layers.Dense(units=10)
])


def loss(model, x, y):
    """Loss function. How bad is the model doing on the entire training set.

    Args:
        model (tf.keras.Model): Keras model.
        x (dict, tf.data.Dataset): Feature tensor.
        y (tuple, list, tf.data.Dataset): Target tensor.

    Returns:
        loss (tf.Tensor): A real value denoting the loss value of this model.
    """
    y_hat = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)


def grad(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor):
    """Estimates the derivative of the loss with respect to the model parameters.

    Returns the derivative of the loss w.r.t. weights and biases. The passing
    this to `optimizer.apply_gradients()`` completes the process of apply
    gradient descent.

    Args:
        model (tf.keras.Model): Keras model.
        inputs (tf.Tensor): Input features.
        targets (tf.Tensor): Target outputs.

    Returns:
        gradients (tfe.GradientTape): Gradients of the model w.r.t it's variables.
    """
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    # Gradient of model w.r.t. it's variables.
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-2)
