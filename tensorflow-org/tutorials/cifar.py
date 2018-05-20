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
    # Command line argument parser.
    parser = argparse.ArgumentParser()

    # Input arguments.
    parser.add_argument('--img_size', type=int, default=28,
                        help="Image size. The default for MNIST data is 28")
    parser.add_argument('--img_depth', type=int, default=1,
                        help="Image channel. The default for MNIST data is 1, "
                             "which signifies image is a Monochrome image.")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of classes to be predicted.")

    # Dataset arguments.
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Mini batch size. Use lower batch size if running on CPU.")
    parser.add_argument('--shuffle_rate', type=int, default=1000,
                        help="Dataset shuffle rate.")
    parser.add_argument('--data_transform_count', type=int, default=5,
                        help="Dataset transform repeat count. "
                             "Use smaller (or 1) if running on CPU")
    parser.add_argument('--feature_col', type=str, default="images",
                        help="Feature column label for tf.feature_column")

    # Estimator arguments.
    parser.add_argument('--save_dir', type=str, default="../../saved/tutorials/mnist",
                        help="Specifies the directory where model data "
                             "(checkpoints) will be saved.")
    parser.add_argument('--logdir', type=str, default="../../saved/tutorials/mnist",
                        help="Specifies the directory where model data "
                             "(checkpoints) will be saved.")
    parser.add_argument('--log_every', type=int, default=50,
                        help="Log specified tensors every ``log_every`` iterations.")

    # Network arguments.
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Kernel size for each convolution. "
                             "default [5, 5]")
    parser.add_argument('--pool_size', type=int, default=2,
                        help="Down-sampling filter size. default [2, 2]")
    parser.add_argument('--filter_conv1', type=int, default=32,
                        help="Size of 1st convolutional filters.")
    parser.add_argument('--filter_conv2', type=int, default=64,
                        help="Size of 2nd convolutional filters.")
    parser.add_argument('--dense_units', type=int, default=1024,
                        help="Number of neurons in the fully connected layer.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout regularization rate (probability that a given "
                             "element will be dropped during training).")

    # Training & optimizer arguments.
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs. Signifies the number of "
                             "times to loop through a complete training iteration. "
                             "Default is `None` meaning that the  model will train "
                             "until the specified number of steps is reached.")
    parser.add_argument('--steps', type=int, default=1000,
                        help="Number of training steps. Represents the number of "
                             "times to loop through a complete mini-batch cycle.")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate for RMSPropOptimizer.")
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help="Decay rate for RMSPropOptimizer.")

    # Parse command line arguments.
    args = parser.parse_args()
    print(args)

    # Start program execution.
    main()
