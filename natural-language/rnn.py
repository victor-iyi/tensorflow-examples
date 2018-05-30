import argparse

import numpy as np
import tensorflow as tf

args = None


def make_one_hot(indices, depth, dtype=np.int32):
    hot = np.zeros(shape=[len(indices), depth], dtype=dtype)

    for i, index in enumerate(indices):
        hot[i:, index] = 1.

    return hot


def make_dataset(features, labels=None, shuffle=False):
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)

    # Transform dataset.
    dataset = dataset.batch(batch_size=args.batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=args.buffer_size)

    return dataset


def load_data(one_hot=False, dataset=True):
    train, test = tf.keras.datasets.mnist.load_data()

    X_train, y_train = train
    X_test, y_test = test

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


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    # Load MNIST dataset as a tf.data.Dataset object.
    train, test = load_data(one_hot=args.one_hot, dataset=True)

    # Create a generic iterator for train & test sets.
    iterator = tf.data.Iterator.from_structure(output_types=train.output_types,
                                               output_shapes=train.output_shapes,
                                               output_classes=train.output_classes)

    # Feature Shape: (batch_size, time_steps, element_size)
    # Labels  Shape: (batch_size, num_classes)
    features, labels = iterator.get_next()

    # Initializes iterator for each train & test dataset.
    # train_iter = iterator.make_initializer(train, name='train_dataset')
    # test_iter = iterator.make_initializer(test, name='test_dataset')

    # Recurrent Network weights & biases.
    # Network weights.
    with tf.name_scope('weights'):
        # Input weights.
        with tf.name_scope('W_x'):
            Wx = tf.get_variable(name='W_x',
                                 shape=(args.element_size, args.hidden_size),
                                 initializer=tf.zeros_initializer())
            variable_summaries(Wx)

        # Recurrent hidden state weights.
        with tf.name_scope('W_h'):
            Wh = tf.get_variable(name='W_h',
                                 shape=(args.hidden_size, args.hidden_size),
                                 initializer=tf.zeros_initializer())
            variable_summaries(Wh)

        # Output layer weights.
        with tf.name_scope('W_o'):
            Wo = tf.get_variable(name='W_o',
                                 shape=(args.hidden_size, args.num_classes),
                                 initializer=tf.zeros_initializer())
            variable_summaries(Wo)

    # Network biases.
    with tf.name_scope('biases'):
        # Hidden state bias.
        with tf.name_scope('b_h'):
            bh = tf.get_variable(name='b_h',
                                 shape=(args.hidden_size),
                                 initializer=tf.zeros_initializer())
            variable_summaries(bh)

        # Output state bias.
        with tf.name_scope('b_o'):
            bo = tf.get_variable(name='b_o',
                                 shape=(args.num_classes),
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            variable_summaries(bo)

    def rnn_step(prev: tf.Tensor, curr: tf.Tensor):
        """Recurrent Neural Net operation at each time step.

        Args:
            prev (tf.Tensor): 
                Previous hidden state.
            curr (tf.Tensor): 
                Input at current time step.

        Returns:
            tf.Tensor -- Current hidden state.
        """
        hidden = tf.matmul(curr, Wx) + tf.matmul(prev, Wh) + bias
        hidden = tf.tanh(hidden)
        return hidden

    # Shape: (time_steps, batch_size, element_size)
    input_trans = tf.transpose(features, perm=[1, 0, 2])

    # Initial hidden state.
    init_hidden = tf.zeros(shape=(args.batch_size, args.hidden_size),
                           name='initial_hidden_state')

    # All hidden state vector across time.
    hidden_states = tf.scan(rnn_step, input_trans,
                            initializer=init_hidden,
                            name='hidden_states')

    def get_outputs(hidden_state):
        return tf.matmul(hidden_state, Wo) + bo

    with tf.name_scope('rnn_ouputs'):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data input arguments.
    parser.add_argument('--element_size', type=int, default=28,
                        help='Element size.')
    parser.add_argument('--time_steps', type=int, default=28,
                        help='Time steps.')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of class labels.')
    parser.add_argument('--one_hot', type=bool, default=True,
                        help='One-hot encode labels?')

    # Network/Model arguments.
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size.')
    parser.add_argument('--dropuot', type=float, default=0.5,
                        help='Droupout Rate.')

    # Data transformation arguments.
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini batch size.')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='Shuffle rate.')

    # Tensorboard arguments.
    parser.add_argument('--logdir', type=str, default='./logs/rnn/logs',
                        help='Tensorboard log directory.')
    parser.add_argument('--save_dir', type=str, default='./saved/rnn/model',
                        help='Model save directory.')

    args = parser.parse_args()

    main()
