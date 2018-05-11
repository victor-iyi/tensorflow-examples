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

import tensorflow as tf
import pandas as pd

# Iris training and testing dataset URL. May change in the future.
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Column names and label names.
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Hyperparameters.
epochs = 500
buffer_size = 1000
batch_size = 32
learning_rate = 1e-2


def _preprocess(dataframe: pd.DataFrame) -> tf.data.Dataset:
    """Pre process dataframe into TensorFlow's dataset object.

    Args:
        dataframe (pd.DataFrame):

    Returns:
        dataset (tf.data.Dataset): Dataset object.
    """
    # Split into features and labels.
    features = dataframe[CSV_COLUMN_NAMES[:-1]].values
    labels = dataframe[CSV_COLUMN_NAMES[-1]].values

    # Create a TensorFlow dataset & apply some pre-processing steps.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)

    # Return the dataset.
    return dataset


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

    # Pre-process dataframe objects.
    train = _preprocess(train_df)
    test = _preprocess(test_df)

    return train, test


class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        # 1st hidden layer
        self.hidden = tf.keras.layers.Dense(units=16, activation='relu', name='hidden')
        self.dropout = tf.keras.layers.Dropout(rate=0.5, name='dropout')
        self.label = tf.keras.layers.Dense(units=1, name='output')

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
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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
    # Minimize the loss.
    return optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())


if __name__ == '__main__':
    # Load training and testing dataset.
    train, test = load_data()

    # Get train data.
    train_iter = train.make_one_shot_iterator()
    X_train, y_train = train_iter.get_next()

    # Get test data.
    test_iter = test.make_one_shot_iterator()
    X_test, y_test = test_iter.get_next()

    # Initialize model & optimizer.
    model = Network()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Calculate the loss & train the model.
    loss = loss_func(model=model, features=X_train, labels=y_train)
    train_op = train_model(optimizer=optimizer, loss=loss)

    # Run within session.
    with tf.Session() as sess:
        # Go through training epochs.
        for epoch in range(epochs):
            # Train the model.
            _, _loss = sess.run([train_op, loss])
            print('\r\tEpoch {} of {}\tLoss {:.3f}'.format(epoch + 1, epochs, _loss),
                  end='')
