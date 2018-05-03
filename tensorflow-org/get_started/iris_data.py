"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: eager-exec.py
    Created on 03 May, 2018 @ 2:11 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    """Download Iris training and testing dataset for Iris.

    Returns:
        [{str}, {str}] - train_path, test_path
    """
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name="Species"):
    """Automatically downloads (if dataset doesn't exist) and loads the Iris dataset.

    Args:
        y_name (str): Column name for the labels.

    Returns:
        [(pd.DataFrame, pd.DataFrame), (pd.DataFrame, pd.DataFrame)] - List of
        tuples containing pd.DataFrame objects. [(train_X, train_y), (test_X, test_y)]
    """
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_X, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_X, test_y = test, test.pop(y_name)

    return (train_X, train_y), (test_X, test_y)


def train_input_fn(features, labels, batch_size):
    """Input function for training.

    Args:
        features (list, tuple): Feature columns of the dataset.
        labels (list): Dataset labels.
        batch_size (int): Mini batch size

    Returns:
        (tf.data.Dataset) - Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels=None, batch_size=64):
    """Input function for evaluation or prediction.

    Args:
        features (dict): Dataset features.
        labels (list, pd.DataFrame, tuple): Dataset labels.
        batch_size (int): Mini batch size. (default = 64)

    Returns:
        (tf.data.Dataset) - Dataset object.
    """
    features = dict(features)
    if labels:
        inputs = (features, labels)
    else:
        inputs = features

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)

    return dataset


# Format which CSV file should assume.
# 4 feature columns of dtype=float & 1 label column of dtype=int
CSV_TYPES = [[0.], [0.], [0.], [0.], [0]]


def _parse_line(line):
    """Process each line of the csv file into features and labels.

    Args:
        line (str):

    Returns:
        (features, labels) - Parsed features and label.
    """
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Features contains 1st 4 columns while labels - last column.
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    labels = fields.pop(CSV_COLUMN_NAMES[-1])

    return features, labels


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the lines. Skip the 1st row.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line & shuffle the dataset.
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset
