import os

import numpy as np


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

    Returns:
        one_hot (np.ndarray): One-hot encoded array.
    """
    hot = np.zeros(shape=(indices.shape[0], depth), dtype=dtype)
    for i, index in enumerate(indices):
        hot[i, index] = 1.
    return hot


if __name__ == '__main__':
    X, y = fake_data(128, size=32, channels=1)
    print('X.shape = {}\ty.shape = {}'.format(X.shape, y.shape))
