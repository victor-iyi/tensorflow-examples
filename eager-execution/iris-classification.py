"""Iris classification using TensorFlow's Eager Execution mode.

  @author
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola

  @project
    File: iris-classification.py
    Created on 06 May, 2018 @ 4:59 PM.

  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import numpy as np

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
# Experimental summary with Eager execution.
from tensorflow.contrib.summary import summary

# Turn on eager execution.
tf.enable_eager_execution()

# Hyperparameters.
buffer_size, batch_size, learning_rate = 1000, 32, 1e-2

"""
# Sci-kit learn utility functions.
from sklearn import datasets, preprocessing, model_selection

# Load the Iris dataset.
data = datasets.load_iris()
TARGET_NAMES = {i: l for i, l in enumerate(data['target_names'])}

print(55 * '-')
print('Feature names: {feature_names}'.format(**data))
print('Target names: {target_names}'.format(**data))
print('Target mapping: {}'.format(TARGET_NAMES))
print(55 * '-', '\n')

# Pre-processing step.
features = preprocessing.MinMaxScaler(feature_range=(-1, 1))
features = features.fit_transform(data['data'])

labels = preprocessing.OneHotEncoder(sparse=False)
labels = labels.fit_transform(data['target'].reshape(-1, 1))

# Split into training and testing sets.
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels,
                                                                    test_size=0.25)
print(55 * '-')
print('X_train.shape = {}\tX_test.shape:{}'.format(X_train.shape, X_test.shape))
print('y_train.shape = {}\ty_test.shape:{}'.format(y_train.shape, y_test.shape))
print(55 * '-', '\n')

# Training set.
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.batch(batch_size=32)

# Testing set.
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
"""


def _parse_line(line: any):
    """Perform pre-processing on each row in the CSV file.

    Args:
        line (any): Each row of the CSV file.

    Returns:
        (features, labels) - Parsed features and labels.
    """
    row = tf.decode_csv(line, record_defaults=[[0.], [0.], [0.], [0.], [0]])
    # Split line into features and labels.
    features = tf.reshape(row[:-1], shape=(4,))
    labels = tf.reshape(row[-1], shape=())
    # Return parsed features & labels.
    return features, labels


def load_data():
    """Load training and testing dataset as a tf.data.Dataset object.

    Returns:
        (tf.data.Dataset, tf.data.Dataset) - train, test
    """
    # Iris training and testing dataset URL. May change in the future.
    TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    # Download dataset if they exist. Otherwise load from disk.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1],
                                        origin=TEST_URL)

    # Train dataset.
    train = tf.data.TextLineDataset(train_path)
    train = train.skip(count=1)
    train = train.map(_parse_line)
    train = train.shuffle(buffer_size=buffer_size)
    train = train.batch(batch_size=batch_size)

    # Test dataset.
    test = tf.data.TextLineDataset(test_path)
    test = test.skip(count=1)
    test = test.map(_parse_line)
    test = test.shuffle(buffer_size=buffer_size)
    test = test.batch(batch_size=batch_size)

    # Return train & test as a tf.data.Dataset object.
    return train, test


# Label names
TARGET_NAMES = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Training and testing set.
train_data, test_data = load_data()


class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        self.hidden_layer = tf.layers.Dense(units=10,
                                            activation=tf.nn.relu,
                                            use_bias=True)
        self.output_layer = tf.layers.Dense(units=3,
                                            activation=None,
                                            use_bias=True)

    def __call__(self, x, **kwargs):
        return self.call(x, **kwargs)

    def call(self, x, **kwargs):
        return self.output_layer(self.hidden_layer(x))

    # Abstract methods.
    def _set_inputs(self, inputs, training=None):
        pass

    def add_variable(self, name, shape, dtype=None, initializer=None,
                     regularizer=None, trainable=True, constraint=None, **kwargs):
        pass

    def add_loss(self, *args, **kwargs):
        pass

    def save(self, filepath, overwrite=True, include_optimizer=True):
        pass


def loss(model: tf.keras.Model, inputs: tf.Tensor, labels: tf.Tensor):
    """Cross entropy loss function.

    Arguments:
        model {tf.keras.Model} -- Instance of tf.Keras.Model
        inputs {tf.Tensor} -- Input features.
        labels {tf.Tensor} -- Output labels.

    Returns:
        tf.Tensor -- Loss value.
    """
    logits = model(inputs)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                             labels=labels)
    return tf.reduce_mean(entropy, name="loss")


def train_step(model: tf.keras.Model, optimizer: tf.train.Optimizer,
               loss: loss, x: tf.Tensor, y: tf.Tensor):
    """Training operation. That is, we minimize the loss function here.

    Arguments:
        model {tf.keras.Model} -- Instance of tf.keras.Model
        optimizer {tf.train.Optimizer} -- Optimizer to be used.
        loss {loss} -- Loss function.
        x {tf.Tensor} -- Input features.
        y {tf.Tensor} -- Output labels.
    """
    optimizer.minimize(loss=lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())


model = Network()
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

# Loop through each batch in the dataset.
for b, (_, y_batch) in enumerate(tfe.Iterator(train_data)):
    y_batch = y_batch.numpy()  # Convert to numpy array.
    print('\nBatch {}'.format(b))
    # Loop through each class.
    for i, class_names in TARGET_NAMES.items():
        # Get the index of class occurrences.
        indices = np.where(y_batch == i)[0]
        per = len(y_batch[indices]) / len(y_batch)
        print('Percentage of class {!r}: {:>.2%}'.format(
            class_names.title(), per))

logdir = '../logs/iris-classification'
epochs = 200
accuracies = np.zeros(epochs)

writer = summary.create_file_writer(logdir=logdir)
with writer.as_default():
    with summary.always_record_summaries():
        # !- Training started.
        print(55 * '-')
        for epoch in range(epochs):
            accuracy = tfe.metrics.Accuracy()
            _loss = 0.
            # Loop over the epochs.
            for X_batch, y_batch in tfe.Iterator(train_data):
                # Save the loss for tensorboard.
                _loss = loss(model, X_batch, y_batch)
                tf.summary.scalar('loss', _loss)

                # Train the network.
                train_step(model, optimizer, loss, X_batch, y_batch)

                # Get accuracy.
                y_pred = tf.argmax(model(X_batch), axis=1,
                                   output_type=tf.int32)
                accuracy(y_pred, y_batch)

            # Get the accuracy & save it for tensorboard.
            acc = accuracy.result()
            tf.summary.scalar('accuracy', acc)

            # Log training progress.
            print('\rEpoch: {:,}\tLoss: {:.4f}\tAcc: {:.2%}'.format(epoch + 1,
                                                                    _loss, acc),
                  end='')
        # !- End training.
        print('\n'.format(55 * '-'))
