"""Classify cifar10 dataset using tf.estimator API.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: cifar.py
     Created on 20 May, 2018 @ 4:51 PM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
  
"""

import argparse

import numpy as np
import tensorflow as tf

# Command line argument.
args = None


def make_one_hot(indices: np.ndarray, depth: int, dtype: np.dtype = np.int32):
    hot = np.zeros(shape=(indices.shape[0], depth), dtype=dtype)

    for i, index in enumerate(indices):
        hot[i, index] = 1.

    return hot


def load_data(one_hot=True):
    # Download dataset.
    train, test = tf.keras.datasets.cifar10.load_data()

    # Split into features & labels.
    X_train, y_train = train
    X_test, y_test = test

    # Pre-process the images.
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    # Convert to one-hot.
    if one_hot:
        y_train = make_one_hot(indices=y_train, depth=10)
        y_test = make_one_hot(indices=y_test, depth=10)

    return (X_train, y_train), (X_test, y_test)


def main():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training arguments.
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help="Learning rate.")

    args = parser.parse_args()

    main()
