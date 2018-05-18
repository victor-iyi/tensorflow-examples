"""A Guide to TF Layers: Building a Convolutional Neural Network.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: mnist.py
     Created on 18 May, 2018 @ 5:26 PM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from __future__ import print_function, absolute_import, division

import argparse

import numpy as np
import tensorflow as tf

# TensorFlow log level.
tf.logging.set_verbosity(tf.logging.INFO)

# Command line arguments.
args = argparse.Namespace


def make_one_hot(indices: np.ndarray, depth: int, dtype: np.dtype = np.int32):
    """Returns a one-hot array.

    Args:
        indices (np.ndarray): Array to be converted.
        depth (int): How many elements per item.
        dtype (np.dtype): Encoded array data type.

    Returns:
        one_hot (np.ndarray): One-hot encoded array.
    """
    hot = np.zeros(shape=(indices.shape[0], depth), dtype=dtype)
    for i, index in enumerate(indices):
        hot[i, index] = 1.
    return hot


def load_data(one_hot=False):
    """Load MNIST dataset.

    Args:
        one_hot (bool):
            Maybe convert labels to one-hot arrays.

    Returns:
        tuple: train, test
    """
    # Maybe download mnist dataset.
    train, test = tf.keras.datasets.mnist.load_data()

    # Split into images & labels.
    X_train, y_train = train
    X_test, y_test = test

    # Release train & test from memory.
    del train, test

    if one_hot:
        y_train = make_one_hot(indices=y_train, depth=10)
        y_test = make_one_hot(indices=y_test, depth=10)

    return (X_train, y_train), (X_test, y_test)


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: tf.estimator.ModeKeys):
    with tf.name_scope('cnn_model'):
        # Input layer.
        input_layer = tf.reshape(tensor=features["x"],
                                 shape=[-1, args.img_size, args.img_size, args.img_depth],
                                 name="input_layer")

        # Convolutional layer #1.
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=args.filter_conv1,
                                 kernel_size=args.kernel_size,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv_layer_1")

        # Pooling layer #1.
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=args.pool_size,
                                        strides=2, name="pool_layer_1")

        # Convolutional layer #2.
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=args.filter_conv2,
                                 kernel_size=args.kernel_size,
                                 activation=tf.nn.relu,
                                 name="conv_layer_2")

        # Pooling layer #2.
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=args.pool_size,
                                        strides=2, name="pool_layer_2")

        # Flatten layer (Prep for fully connected layers).
        flatten = tf.layers.flatten(inputs=pool2, name="flatten_layer")

        # Fully Connected or Dense layer #1.
        dense = tf.layers.dense(inputs=flatten,
                                units=args.dense_units,
                                activation=tf.nn.relu,
                                name="fully_connected_layer")

        # Dropout for regularization.
        dropout = tf.layers.dropout(inputs=dense,
                                    rate=args.dropout,
                                    training=mode == tf.estimator.ModeKeys.TRAIN,
                                    name="dropout")

        # Logits layer.
        logits = tf.layers.dense(inputs=dropout,
                                 units=args.num_classes,
                                 name="logits")

        # Predictions.
        with tf.name_scope("predictions"):
            predictions = {
                "classes": tf.argmax(input=logits, axis=1, name="classes"),
                "prob": tf.nn.softmax(logits=logits, name="probabilities")
            }

        # If mode=tf.estimator.ModeKeys.PREDICT, return the predictions.
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate loss (for both TRAIN & EVAL modes).
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=logits,
                                               reduction=tf.losses.Reduction.MEAN,
                                               name="loss")


def main():
    # Load MNIST dataset.
    train, test = load_data(one_hot=True)

    # Split into image & labels.
    X_train, y_train = train
    X_test, y_test = test


if __name__ == '__main__':
    # Command line argument parser.
    parser = argparse.ArgumentParser()

    # Input arguments.
    parser.add_argument('--img_size', type=int, default=28,
                        help="Image size. The default for MNIST data is 28")
    parser.add_argument('--img_depth', type=int, default=1,
                        help="Image channel. The default for MNIST data is 1,"
                             " which signifies image is a Monochrome image.")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of classes to be predicted.")

    # Network arguments.
    parser.add_argument('--kernel_size', type=list, default=[5, 5],
                        help="Kernel size for each convolution.")
    parser.add_argument('--pool_size', type=list, default=[2, 2],
                        help="Down-sampling filter size.")
    parser.add_argument('--filter_conv1', type=int, default=32,
                        help="Size of 1st convolutional filters.")
    parser.add_argument('--filter_conv2', type=int, default=64,
                        help="Size of 2nd convolutional filters.")
    parser.add_argument('--dense_units', type=int, default=1024,
                        help="Number of neurons in the first (and only) fully"
                             " connected layer.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout regularization rate (probability that a given"
                             " element will be dropped during training).")

    # Checkpoints & savers.
    parser.add_argument('--save_dir', type=str, default="../../saved/tutorials/mnist",
                        help="Path to save checkpoint files.")

    # Parse command line arguments.
    args = parser.parse_args()

    # Start program execution.
    main()
