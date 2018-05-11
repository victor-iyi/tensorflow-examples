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

    # Split into features and labels.
    X_train, y_train = train_df[CSV_COLUMN_NAMES[:-1]].values, train_df[CSV_COLUMN_NAMES[-1]].values
    X_test, y_test = test_df[CSV_COLUMN_NAMES[:-1]].values, test_df[CSV_COLUMN_NAMES[-1]].values

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
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
    logits = model(features)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss, name="loss")


def train_op(model: tf.keras.Model, optimizer: tf.train.Optimizer,
             loss_func: any, x: tf.Tensor, y: tf.Tensor) -> None:
    # Calculate the loss.
    loss = loss_func(model, x, y)

    # Minimize the loss.
    optimizer.minimize(loss=loss,
                       global_step=tf.train.get_or_create_global_step())


if __name__ == '__main__':
    train, test = load_data()

    iterator = train.make_initializable_iterator()
    features, labels = iterator.get_next()
