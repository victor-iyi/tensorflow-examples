"""A simple Linear Classifier to find `W` and `b`.

  @author
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola

  @project
    File: simple.py
    Created on 13 May, 2018 @ 11:32 AM.

  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()


def load_data(n: int=1000, W: int=3, b: int=2):
    noise = tf.random_normal(shape=[n])
    X = tf.random_normal(shape=[n])
    y = X * W + b + noise
    return X, y


def prediction(X: tf.Tensor, W: tf.Tensor, b: tf.Tensor):
    return X * W + b


def loss_func(y_hat: tf.Tensor, y: tf.Tensor):
    # y_hat = prediction(X, W, b)
    return tf.reduce_mean(tf.square(y_hat - y))


def grad(X, y, W, b):
    with tfe.GradientTape() as tape:
        y_hat = prediction(X, W, b)
        loss = loss_func(y_hat=y_hat, y=y)

    # Compute gradient of `loss` w.r.t. `W` & `b`.
    return tape.gradient(target=loss, sources=[W, b])


def visualize(X, y, y_pred=None):
    fig = plt.figure()

    plt.scatter(X.numpy(), y.numpy(), c='b', s=50, marker='*')
    if y_pred is not None:
        plt.plot(X.numpy(), y_pred.numpy(), c='r', linewidth=5)

    plt.show()
    del fig


if __name__ == '__main__':

    num_samples, data_W, data_b = 1000, 3, 2

    # Load training dataset.
    X, y = load_data(n=num_samples, W=data_W, b=data_b)

    # Model variables.
    W = tfe.Variable(tf.zeros(shape=()), name="weights")
    b = tfe.Variable(tf.zeros(shape=()), name="biases")

    epochs = 500
    learning_rate = 1e-2

    for epoch in range(epochs):
        dW, db = grad(X, y, W, b)

        # Update W & b.
        W.assign_sub(learning_rate * dW)
        b.assign_sub(learning_rate * db)
        loss = loss_func(prediction(X, W, b), y)

        print(('\rEpoch {:,}\tLoss {:3f}\tW = {:.2f}'
               '\tb={:.2f}').format(epoch + 1, loss.numpy(),
                                    W.numpy(), b.numpy()),
              end='')

    print('\n\nOriginal: W = {}\tb = {}'.format(data_W, data_b))
    print('Final: W = {:.0f}\tb = {:.0f}'.format(W.numpy(), b.numpy()))

    X, y = load_data(n=100, W=3, b=2)
    y_pred = prediction(X, W, b)

    visualize(X, y, y_pred)
