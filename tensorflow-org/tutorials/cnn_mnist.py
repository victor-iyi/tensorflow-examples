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

tf.logging.set_verbosity(tf.logging.INFO)

args = None


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


def load_data(one_hot=True):
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


def main():
    train, test = load_data()
    # print(train, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="../../saved/tutorials/mnist",
                        help="Path to save checkpoint files.")

    args = parser.parse_args()

    main()
