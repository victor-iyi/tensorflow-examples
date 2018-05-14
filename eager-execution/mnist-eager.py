"""MNIST classification using TensorFlow's Eager execution mode.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: mnist-eager.py
     Created on 14 May, 2018 @ 5:49 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.

"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def load_data():
    """Load the MNIST dataset into train & testing set.

    Returns:
        train, test (tuple): Training and testing set.
    """
    train, test = tf.keras.datasets.mnist.load_data()

    return train, test


def pre_process(features, labels):
    """Flatten images & one-hot encode labels.

    Arguments:
        features {tf.Tensor} -- Dataset images.
        labels {tf.Tensor} -- Dataset labels.

    Returns:
        {(tf.Tensor, tf.Tensor)} -- features, labels
    """
    # Reshaping image to fit the model.
    img_size_flat = np.prod(features.shape[1:])
    features = features.reshape((-1, img_size_flat))

    # One-hot encoding.
    num_classes = len(np.unique(labels))
    labels = tf.one_hot(indices=labels, depth=num_classes)

    # Return processed features & labels.
    return features, labels


def process_data(features, labels, batch_size=64, buffer_size=1000):

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.hidden = tf.keras.layers.Conv2D(filters=5,
                                             kernel_size=2,
                                             activation='relu')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=512)
        self.fc2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # Conv & Pooling layer
        result = self.pool(self.hidden(inputs))
        # Flatten layer.
        result = self.flatten(result)
        # Fully connected layers.
        result = self.fc2(self.fc1(result))
        # Output prediction.
        return result


def main():
    # Logging split.
    print('\n{}'.format(60 * '-'))

    # Load data & split into training & testing sets.
    train, test = load_data()
    X_train, y_train = train
    X_test, y_test = test

    # Number of training/testing samples.
    n_train, n_test = y_train.shape[0], y_test.shape[0]
    print(f'{n_train:,} train samples\t &'
          f'\t{n_test:,} testing samples')

    # Image dimensions.
    img_shape = X_train.shape[1:]
    img_size, img_depth = img_shape[0], 1
    img_size_flat = img_size * img_size * img_depth
    print(f'Image  = Shape: {img_shape}\tSize: {img_size}'
          f'\tDepth: {img_depth}\tFlat: {img_size_flat}')

    # Output dimensions.
    classes = np.unique(y_train)
    num_classes = len(classes)
    print(f'Labels = Classes: {classes}\tLength: {num_classes}')

    # Logging split.
    print('{}\n'.format(60 * '-'))

    X_train, y_train = pre_process(X_train, y_train)
    X_test, y_test = pre_process(X_test, y_test)

    data_train = process_data(X_train, y_train,
                              batch_size=128, buffer_size=1000)
    data_test = process_data(X_test, y_test,
                             batch_size=68, buffer_size=1000)

    # for batch, (X, y) in enumerate(data_train):
    #     pass


if __name__ == '__main__':
    main()
