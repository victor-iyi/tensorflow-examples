"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: iris_classification.py
    Created on 06 May, 2018 @ 10:23 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

# Turn on eager execution mode.
tf.enable_eager_execution()


def _parse_line(line):
    rows = tf.decode_csv(line, record_defaults=[[0.], [0.], [0.], [0.], [0]])

    features = tf.reshape(rows[:-1], shape=(4,))
    labels = tf.reshape(rows[-1], shape=())

    return features, labels


def process(path: str, batch_size: int = 32, buffer_size: int = 1000):
    """Pre process a dataset. Given the file path to a CSV file.

    Args:
        path (str): Path to a CSV file.
        batch_size (int): Representing the number of consecutive
                elements of this dataset to combine in a single batch.
        buffer_size (int): Representing the number of elements from this
            dataset from which the new dataset will sample.

    Returns:
        dataset (tf.data.Dataset):
            Processed dataset object..
    """
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.skip(count=1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset


def load_data(**kwargs):
    """Load the Iris dataset into a `tf.data.Dataset` object.

    Args:
        **kwargs ():
            batch_size (int): Representing the number of consecutive
                elements of this dataset to combine in a single batch.
            buffer_size (int): Representing the number of elements from this
                dataset from which the new dataset will sample.

    Returns:
        (train, test) (tf.data.Dataset, tf.data.Dataset):
            Pre-processed Training and testing set.
    """
    # Dataset URL.
    TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    # Download dataset if it doesn't exist, otherwise, load from disk.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1], origin=TRAIN_URL)
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1], origin=TEST_URL)

    # Read train & test set into Dataset objects.
    train = process(train_path, **kwargs)
    test = process(test_path, **kwargs)

    # Return the parsed training and testing set.
    return train, test


train_data, test_data = load_data()


class Model(tf.keras.Model):
    """A single layer feed forward neural network with dropout.

    Usage:
    ```python
    >>> import tensorflow as tf
    >>>
    >>> # Iris features (measurements).
    >>> features = tf.convert_to_tensor({
    ...    [5.1, 3.3, 1.7, 0.5],
    ...    [5.9, 3.0, 4.2, 1.5],
    ...    [6.9, 3.1, 5.4, 2.1],
    ... })
    >>> # Class label mappings.
    >>> class_ids = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    >>> # Define our model.
    >>> model = Model()
    >>> # Make a prediction.
    >>> pred = model(features)
    >>> # Loop through prediction and print corresponding class names.
    >>> for i, logits in enumerate(pred):
    ...    class_idx = tf.argmax(logits).numpy()
    ...    name = class_ids[class_idx]
    ...    print('Example {} prediction: {}'.format(i, name))
    ```
    """

    def __init__(self):
        """Initializes a single layer feed forward neural network."""
        super(Model, self).__init__()

        # Network definition.
        self.hidden_layer = tf.keras.layers.Dense(units=10, activation='relu', name='hidden_layer')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.output_layer = tf.keras.layers.Dense(units=3, activation=None, name='output_layer')

    def __call__(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

    def call(self, x, **kwargs):
        hidden = self.hidden_layer(x)
        dropout = self.dropout(hidden)
        return self.output_layer(dropout)

    def _set_inputs(self, inputs, training=None):
        pass

    def add_variable(self, name, shape, dtype=None, initializer=None,
                     regularizer=None, trainable=True, constraint=None,
                     **kwargs):
        pass

    def add_loss(self, *args, **kwargs):
        pass

    def save(self, filepath, overwrite=True, include_optimizer=True):
        pass


model = Model()


def loss(model: tf.keras.Model, inputs: tf.Tensor, labels: tf.Tensor, sparse: bool = True):
    """Computes the loss function of a sparse label.

    Args:
        model (tf.keras.Model): Instance of `tf.keras.Model`.
        inputs (tf.Tensor): Dataset's input features.
        labels (tf.Tensor): Dataset true labels.
        sparse (bool): False if labels are not one-hot encoded.

    Returns:
        loss (tf.Tensor): Entropy loss.
    """
    logits = model(inputs)
    if sparse:
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=labels)
    else:
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                             labels=labels)
    return tf.reduce_mean(entropy, name="loss")


optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)


def train_step(model: tf.keras.Model, optimizer: tf.train.Optimizer,
               loss_func: loss, inputs: tf.Tensor, labels: tf.Tensor, **kwargs):
    """Kicks off training for a given model.

    Args:
        model (tf.keras.Model):
        optimizer (tf.train.Optimizer):
        loss_func (loss): Loss function.
        inputs (tf.Tensor): Dataset's input features.
        labels (tf.Tensor): Dataset true labels.

    Keyword Args:
            sparse (bool): False if labels are not one-hot encoded.

    Returns:
        An Operation that updates the variables in `var_list`.  If `global_step`
              was not `None`, that operation also increments `global_step`.
    """
    loss = loss_func(model, inputs, labels, **kwargs)
    return optimizer.minimize(loss=loss,
                              global_step=tf.train.get_or_create_global_step)


train_accuracy = tfe.metrics.Accuracy()
#
# epochs = 10000
# for epoch in range(epochs):
#     # Loop through each data batches.
#     for X_batch, y_batch in tfe.Iterator(train_data):
#         # Run the training step.
#         train_step(model, optimizer, loss, X_batch, y_batch)
#
#         # Estimate accuracy.
#         y_pred = tf.argmax(model(X_batch), axis=1, output_type=tf.int32)
#         train_acc = train_accuracy(y_pred, y_batch)
#
#     print('\rEpoch {:03d,}\t Accuracy: {:.3%}'.format(epoch, train_accuracy.result()),
#           end='')
