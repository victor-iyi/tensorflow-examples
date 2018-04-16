import numpy as np
import matplotlib.pyplot as plt


def fake_data(n, size=28, channels=1):
    """Generate image like fake dataset.

    Arguments:
        n {int} -- Size of dataset. This represent how many
            data points to be generated.

    Keyword Arguments:
        size {int} -- widthxheight of the generated data.
            Note that the data will have the same hieght and width
            for simplicity. (default: {28})

        channels {int} -- How many color channels. (default: {1})

    Returns:
        [list of ndarray] -- image and it's corresponding label.
    """

    data = np.ndarray(shape=[n, size, size, channels],
                      dtype=np.float32)
    labels = np.zeros(shape=(n,), dtype=np.int64)

    for i in range(n):
        label = i % 2
        data[i, :, :, 0] = label - 0.5
        labels[i] = label

    return data, labels


if __name__ == '__main__':
    X, y = fake_data(128, size=32, channels=1)
    print('X.shape = {}\ty.shape = {}'.format(X.shape, y.shape))
