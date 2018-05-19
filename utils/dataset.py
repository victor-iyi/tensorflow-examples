import os

import numpy as np
import tensorflow as tf


def fake_data(n, size=28, channels=1):
    """Generate image like fake dataset.

    Arguments:
        n {int} -- Size of dataset. This represent how many
            data points to be generated.

    Keyword Arguments:
        size {int} -- width x height of the generated data.
            Note that the data will have the same height and width
            for simplicity. (default: {28})

        channels {int} -- How many color channels. (default: {1})

    Returns:
        [{list}, {ndarray}] -- image and it's corresponding label.

    Example:
        >>> num_examples = 128
        >>> X, y = fake_data(num_examples, size=32, channels=3)
        >>> X.shape
        (128, 32, 32, 3)
        >>> y.shape
        (128,)
    """
    data = np.ndarray(shape=[n, size, size, channels], dtype=np.float32)
    labels = np.zeros(shape=(n,), dtype=np.int64)

    for i in range(n):
        label = i % 2
        data[i, :, :, 0] = label - 0.5
        labels[i] = label

    return data, labels


def gen_data(file: str, max_files: int = 50):
    """Generate a Python code dataset.

    Using the builtin standard libraries as a reference.
    It works by combining {max} number of the reference files.

    Arguments:
        file {str} -- Name of the file to be written into.

    Keyword Arguments:
        max_files {int} -- Maximum number of files to join. (default: {50})

    Example:
        >>> data_path = 'datasets/pycode/input.txt'
        >>> gen_data(data_path, max_files=20)
        >>> import os.path
        >>> os.path.isfile(data_path)
        True
    """
    PYTHON_HOME = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/'

    # Clean the directory.
    if os.path.isfile(file):
        import shutil
        shutil.rmtree(os.path.dirname(file))

    # Create data directories.
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

    counter = 0
    for (root, _, files) in os.walk(PYTHON_HOME):
        files = [os.path.join(root, f) for f in files if f.endswith('.py')]
        for _file in files:
            with open(file, mode='a', encoding='utf-8') as handle:
                try:
                    code = open(_file, 'r', encoding='utf-8').read()
                    handle.write("{}\n\n".format(code))
                except Exception as e:
                    print('EXCEPTION: {}'.format(e))

        counter += 1

        # End loop if reached specified maximum number of files.
        if counter > max_files:
            break


def make_one_hot(indices: np.ndarray, depth: int, dtype: np.dtype = np.int32):
    """Returns a one-hot array.

    Args:
        indices (np.ndarray): Array to be converted.
        depth (int): How many elements per item.
        dtype (np.dtype): Encoded array data type.

    Examples:
        ```python
        >>> y = np.random.randint(low=0, high=10, size=(5,))
        >>> print(y)
        [4 9 6 7 5]
        >>> y_hot = make_one_hot(indices=y, depth=10)
        >>> print(y_hot)
        [[0 0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0 1 0 0]
         [0 0 0 0 0 1 0 0 0 0]]
        ```

    Returns:
        one_hot (np.ndarray): One-hot encoded array.
    """
    hot = np.zeros(shape=(indices.shape[0], depth), dtype=dtype)
    for i, index in enumerate(indices):
        hot[i, index] = 1.
    return hot


def load_data(one_hot=False):
    """Load MNIST dataset.

    Args:
        one_hot (bool):
            Maybe convert labels to one-hot arrays.

    Examples:
        ```python
        >>> train, test = load_data(one_hot=True)
        >>> X_train, y_train = train
        >>> X_test, y_test = test
        >>> print('Train: images = {}\t labels = {}'.format(X_train.shape, y_train.shape))
        Train: images = (60000, 28, 28)	 labels = (60000, 10)

        >>> print('Test: images = {}\t labels = {}'.format(X_test.shape, y_test.shape))
        Test: images = (10000, 28, 28)	 labels = (10000, 10)

        ```

    Returns:
        tuple: train, test
    """
    # Maybe download mnist dataset.
    train, test = tf.keras.datasets.mnist.load_data()

    # Split into images & labels.
    X_train, y_train = train
    X_test, y_test = test

    # Release train & test from memory.
    del train, test

    # Convert images to np.float32
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    # Convert labels to one hot vectors.
    if one_hot:
        y_train = make_one_hot(indices=y_train, depth=10)
        y_test = make_one_hot(indices=y_test, depth=10)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    # fake_data demo
    # X, y = fake_data(128, size=32, channels=1)
    # print('X.shape = {}\ty.shape = {}'.format(X.shape, y.shape))

    # load_data demo
    # train, test = load_data(one_hot=True)
    # X_train, y_train = train
    # X_test, y_test = test
    # print('Train: images = {}\t labels = {}'.format(X_train.shape, y_train.shape))
    #
    # print('Test: images = {}\t labels = {}'.format(X_test.shape, y_test.shape))

    # make_one_hot demo
    y = np.random.randint(low=0, high=10, size=(5,))
    y_hot = make_one_hot(indices=y, depth=10)
    print(y)
    print(y_hot)
