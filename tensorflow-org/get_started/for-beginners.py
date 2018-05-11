"""Reference from https://tensorflow.org/develop/getting_started/for_beginners.html

  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: for-beginners.py
    Created on 11 May, 2018 @ 3:42 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import numpy as np
import pandas as pd

import tensorflow as tf

# Iris training and testing dataset URL. May change in the future.
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Column names and label names.
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Hyperparameters.
epochs = 2000
buffer_size = 500
batch_size = 32
learning_rate = 1e-2


def one_hot(values: np.array, dtype: np.dtype = np.int32) -> np.array:
    """Returns the one-hot encoding of `values` with data type `dtype`.

    Args:
        values (np.array): Array to be encoded.
        dtype (np.dtype): Data type of the encoded array.

    Returns:
        hot (np.array): One-hot encoding of `values`.
    """
    u, idx = np.unique(values, return_inverse=True)
    shape = (values.shape[0], u.shape[0])
    hot = np.zeros(shape=shape, dtype=dtype)
    for i, h in zip(idx, hot):
        h[i] = 1.
    return hot


def preprocess(dataframe: pd.DataFrame) -> tuple:
    """Pre process dataframe into TensorFlow's dataset object.

    Args:
        dataframe (pd.DataFrame):

    Returns:
        dataset (tf.data.Dataset): Dataset object.
    """
    # Split into features and one-hot labels.
    features = dataframe[CSV_COLUMN_NAMES[:-1]].values
    labels = one_hot(dataframe[CSV_COLUMN_NAMES[-1]].values)

    return features, labels


def load_data() -> tuple:
    """Loads the Iris dataset from web (if not on disk).

    Returns:
        train, test(tf.data.Dataset, tf.data.Dataset):
            Training and testing tf.data.Dataset objects.
    """
    # Download the dataset if it doesn't exist.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1],
                                        origin=TEST_URL)

    # Load the downloaded CSV into a pd.DataFrame object.
    train_df = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, skiprows=1)
    test_df = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, skiprows=1)

    # Train features and labels.
    train = preprocess(dataframe=train_df)
    test = preprocess(dataframe=test_df)

    return train, test


class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        # 1st hidden layer
        self.hidden = tf.keras.layers.Dense(units=16, activation='relu', name='hidden')
        self.dropout = tf.keras.layers.Dropout(rate=0.5, name='dropout')
        self.label = tf.keras.layers.Dense(units=3, name='output')

    def __call__(self, features, **kwargs):
        return self.call(features, **kwargs)

    def call(self, x, **kwargs):
        hidden = self.dropout(self.hidden(x))
        return self.label(hidden)

    def add_loss(self, *args, **kwargs):
        pass

    def save(self, filepath, overwrite=True, include_optimizer=True):
        pass

    def add_variable(self, name, shape, dtype=None, initializer=None,
                     regularizer=None, trainable=True, constraint=None, **kwargs):
        pass

    def _set_inputs(self, inputs, training=None):
        pass


def loss_func(model: tf.keras.Model, features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Loss function. Calculate how bad the model is doing on the entire sample.

    Args:
        model (tf.keras.Model): Instance of Keras' model.
        features (tf.Tensor): Input features
        labels (tf.Tensor): Output labels.

    Returns:
        loss (tf.Tensor):
            Mean of the computed loss
    """
    logits = model(features)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    return tf.reduce_mean(loss, name="loss")


def train_model(optimizer: tf.train.Optimizer, loss: tf.Tensor):
    """Minimize the loss with respect to the model variables.

    Args:
        optimizer (tf.train.Optimizer):
        loss (tf.Tensor): Loss value as defined by a loss function..

    Returns:
        An Operation that updates the variables in `var_list`
        & also increments `global_step`.
    """
    return optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())


def main():
    # Load training and testing dataset.
    train_data, test_data = load_data()

    # Dataset dimensions.
    n_train, n_test = train_data[0].shape[0], test_data[1].shape[0]
    n_features, n_labels = test_data[0].shape[1], test_data[1].shape[1]

    print(('\nNumber of training samples: {:,}\nNumber of testing samples: {:,}'
           '\nFeatures: {:,}\tLabels: {:,}\n').format(n_train, n_test, n_features, n_labels))

    # Features and labels placeholder.
    X_plhd = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    y_plhd = tf.placeholder(dtype=tf.int32, shape=[None, 3])

    # Train & test dataset objects.
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(X_plhd, y_plhd))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Train & test iterable objects.
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    # Initialize model & optimizer.
    model = Network()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Calculate the loss & train the model.
    loss = loss_func(model=model, features=features, labels=labels)
    train_op = train_model(optimizer=optimizer, loss=loss)

    # Retrieve the optimizer's global step.
    global_step = tf.train.get_global_step()

    # TODO(victor-iyiola): Estimate model's accuracy on test set.
    logits = model(features)
    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Running the computational graph (on tf.get_default_graph).
    with tf.Session() as sess:
        # Initialize global variables.
        sess.run(tf.global_variables_initializer())
        feed_dict = {X_plhd: train_data[0], y_plhd: train_data[1]}

        for epoch in range(epochs):
            sess.run(iterator.initializer, feed_dict=feed_dict)

            # Go through mini-batch.
            for batch in range(n_train // batch_size):
                # Train the network.
                _, _loss, _global_step, _acc = sess.run([train_op, loss, global_step, accuracy])

                # Log training progress.
                print(('\rEpoch {:,} of {:,}\tGlobal step: {:,}'
                       '\tLoss: {:.3f}\tAcc: {:.2%}')
                      .format(epoch + 1, epochs, _global_step, _loss, _acc), end='')


if __name__ == '__main__':
    main()
