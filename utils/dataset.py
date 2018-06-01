import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import batch_and_drop_remainder


def make_dataset(features: np.ndarray, labels: np.ndarray = None, shuffle: bool = False):
    """Converts features and labels into a tf.data.Dataset object.

    Arguments:
        features {np.ndarray} -- NumPy Array containing features.

    Keyword Arguments:
        labels {np.ndarray} -- NumPy array containing  labels. (default: {None})
        shuffle {bool} -- Shuffle the dataset? (default: {False})

    Returns:
        tf.data.Dataset -- Dataset object.
    """

    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)

    # Transform dataset.
    # dataset = dataset.batch(batch_size=args.batch_size)
    dataset = dataset.apply(batch_and_drop_remainder(args.batch_size))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=args.buffer_size)

    return dataset


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
    # Length of the array.
    n = len(indices)

    # One-hot encoding.
    hot = np.zeros(shape=(n, depth), dtype=dtype)
    hot[range(n), indices] = 1.

    return hot


def load_data(one_hot: bool=False, dataset: bool=False):
    """Load MNIST dataset into an optional tf.data.Dataset object.

    Args:
        one_hot (bool):
            Maybe convert labels to one-hot arrays.
        dataset (bool):
            Should return tf.data.Dataset object. (default: {False})

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
        ```python
        >>> train, test = load_data(one_hot=True, dataset=True)
        >>> iterator = tf.data.Iterator.from_structure(
                            output_types=train.output_types,
                            output_shapes=train.output_shapes,
                            output_classes=train.output_classes
                        )
        >>> features, labels = iterator.get_next()
        >>> # model.train(features, labels)

        ```

    Returns:
        tuple: Train and test dataset splits.
    """
    train, test = tf.keras.datasets.mnist.load_data()

    X_train, y_train = train
    X_test, y_test = test

    del train, test

    # Change dtype of features.
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    if one_hot:
        y_train = make_one_hot(y_train, depth=args.num_classes)
        y_test = make_one_hot(y_test, depth=args.num_classes)

    if dataset is False:
        return (X_train, y_train), (X_test, y_test)

    train_data = make_dataset(X_train, y_train, shuffle=True)
    test_data = make_dataset(X_test, y_test, shuffle=False)

    return train_data, test_data


def odd_even_sequences(n: int, seq_len: int=6, pad: bool=True, **kwargs):
    """Generate odd & even number sequences (as English words).

    Args:
        n (int): Number of sequential data to generate.
        seq_len (int): Sequence lengths. (default {6})
        pad (bool): Apply zero padding to varying sequences (default {True})

    Keyword Args:
         min_len (int): Minimum sequence length. (default {3})
         max_len (int): Maximum sequence length. (default {7})
         return_sequences (bool): Should return the sequence lengths
            alongside the data. (default {True})

    Examples:
        ```python
        >>> # With padding.
        >>> odd_seq, even_seq = odd_even_sequences(100, seq_len=5)
        >>> print(odd_seq[:3])
        ['Nine Seven Seven One Nine', 'Seven Nine Nine PAD PAD', 'Nine Three Nine Nine PAD']
        >>> print(even_seq[:3])
        ['Four Eight Six Two Two', 'Four Four Six PAD PAD', 'Six Two Two Two PAD']
        ```

        ```python
        >>> # With out padding.
        >>> (odd_seq, even_seq), seqs = odd_even_sequences(100, pad=False, return_sequences=True)
        >>> print(odd_seq[:3])
        ['Five One Three Three Five', 'Nine One Three Nine', 'Three Three Seven One One Three']
        >>> print(even_seq[:3])
        ['Four Six Four Eight Eight', 'Six Two Two Four', 'Eight Six Four Eight Two Two']
        >>> print(seqs[:3])
        [5, 4, 6]
        ```

    Returns:
        tuple -- (evens, odds).
            (evens, odds), seq_lens if `return_sequences` == True.
    """
    # Extract keyword arguments.
    min_len = kwargs.get('min_len') or 3
    max_len = kwargs.get('max_len') or 7
    return_sequences = kwargs.get('return_sequences') or False

    # Assert minimum & maximum sequence lengths.
    assert min_len > 1 and max_len < 10, 'Sequence Length Assertion: Min of 1 & Max of 7'

    DIGIT_MAP = {
        0: 'PAD', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
        5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'
    }

    evens, odds, seq_lens = [], [], []

    for i in range(n):
        rand_seq_len = np.random.choice(range(min_len, max_len))
        seq_lens.append(rand_seq_len)
        odd = np.random.choice(range(1, 10, 2), size=rand_seq_len)
        even = np.random.choice(range(2, 10, 2), size=rand_seq_len)

        # Sequence padding.
        if pad and rand_seq_len < seq_len:
            odd = np.append(odd, [0] * (seq_len - rand_seq_len))
            even = np.append(even, [0] * (seq_len - rand_seq_len))

        # Random numbered words.
        odds.append(' '.join([DIGIT_MAP[r] for r in odd]))
        evens.append(' '.join([DIGIT_MAP[r] for r in even]))

    return (odds, evens), seq_lens if return_sequences else (odds, evens)


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
