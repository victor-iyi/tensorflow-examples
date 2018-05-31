"""Recurrent Neural Network Architecture Simple Implementation.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: rnn.py
     Created on 30 May, 2018 @ 8:20 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import argparse

import numpy as np
import tensorflow as tf

# Command line arguments.
args = None


def make_one_hot(indices: np.ndarray, depth: int, dtype: np.dtype = np.int32):
    """Convert a list of numbered indices to corresponding one-hot vectors.

    Arguments:
        indices {np.ndarray} -- Numbered indices of `on` signal.
        depth {int} -- Length of each encoded vector.

    Keyword Arguments:
        dtype {np.dtype} -- Data type. (default: {np.int32})

    Returns:
        np.ndarray -- One-hot vector.
    """
    hot = np.zeros(shape=[len(indices), depth], dtype=dtype)

    for i, index in enumerate(indices):
        hot[i:, index] = 1.

    return hot


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
    dataset = dataset.batch(batch_size=args.batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=args.buffer_size)

    return dataset


def load_data(one_hot=False, dataset=True):
    """Load MNIST dataset into an optional tf.data.Dataset object.

    Keyword Arguments:
        one_hot {bool} -- Should one-hot encode labels. (default: {False})
        dataset {bool} -- Should return tf.data.Dataset object. (default: {True})

    Returns:
        tuple -- Train and test dataset splits.
    """
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


def variable_summaries(var: tf.Tensor):
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
    train_iter = iterator.make_initializer(train, name='train_dataset')
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
                                 shape=[args.hidden_size],
                                 initializer=tf.zeros_initializer())
            variable_summaries(bh)

        # Output state bias.
        with tf.name_scope('b_o'):
            bo = tf.get_variable(name='b_o',
                                 shape=[args.num_classes],
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
        hidden = tf.matmul(curr, Wx) + tf.matmul(prev, Wh) + bh
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

    def get_outputs(hidden_state: tf.Tensor):
        """Apply vanilla linear function with no activation.

        Arguments:
            hidden_state (tf.Tensor): Hidden layer.

        Returns:
            tf.Tensor -- Output logits.
        """
        return tf.matmul(hidden_state, Wo) + bo

    with tf.name_scope('rnn_outputs'):
        rnn_outputs = tf.map_fn(get_outputs, hidden_states)
        logits = rnn_outputs[-1]
        outputs = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits,
                                               reduction=tf.losses.Reduction.MEAN)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step,
                                      name='train_op')

    with tf.name_scope('accuracy'):
        y_pred = tf.argmax(outputs, axis=1)
        y_true = tf.argmax(labels, axis=1)
        # Correct predictions.
        correct = tf.equal(y_pred, y_true, name='correct')
        # Accuracy.
        accuracy = tf.reduce_mean(tf.cast(correct, tf.int32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # Merge all Tensorboard summaries.
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # Saver & Summary writer.
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logdir=args.logdir, graph=sess.graph)

        init = tf.global_variables_initializer()

        # Restore last checkpoint if it exists.
        if tf.gfile.Exists(args.save_dir):
            try:
                ckpt_path = tf.train.latest_checkpoint(args.save_dir)
                saver.restore(sess, save_path=ckpt_path)
                print('INFO: Restored checkpoint from {}'.format(ckpt_path))
            except Exception as e:
                print('WARN: Could not load checkpoint. {}'.format(e))
                sess.run(init)
        else:
            tf.gfile.MakeDirs(args.save_dir)
            print('INFO: No checkpoints found. Creating checkpoint @ {}'
                  .format(args.save_dir))
            sess.run(init)

        # Reset iterator initializer.
        sess.run(train_iter)

        # Each training epochs.
        for epoch in range(args.epochs):
            try:
                while True:
                    try:
                        # Train the network.
                        _, _step, _loss, _acc, summary = sess.run([train_op, global_step,
                                                                   loss, accuracy, merged])
                        # Log training to tensorboard.
                        writer.add_summary(summary=summary, global_step=_step)

                        print('\rEpoch: {:,}\tStep: {:,}\tAcc: {:.2%}\tLoss: {:.3f}'
                              .format(epoch + 1, _step, _acc, _loss), end='')

                        # Save model.
                        if _step % args.save_every == 0:
                            print('\nSaving model to {}'.format(args.save_dir))
                            saver.save(sess=sess, save_path=args.save_dir,
                                       global_step=global_step)
                    except tf.errors.OutOfRangeError:
                        # Batch ended.
                        print('\nEnd batch!')
                        # Re-initialize the train dataset iterator.
                        sess.run(train_iter)
                        break
            except KeyboardInterrupt:
                print('\nTraining interrupted by user!')

                # Save learned model.
                print('Saving model to {}'.format(args.save_dir))
                saver.save(sess=sess, save_path=args.save_dir,
                           global_step=global_step)

                # End training.
                break


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
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout Rate.')

    # Data transformation arguments.
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Mini batch size.')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='Shuffle rate.')

    # Training arguments.
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Optimizer\'s learning rate.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training iteration/epochs.')
    parser.add_argument('--save_dir', type=str, default='../saved/rnn/model',
                        help='Model save directory.')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save model every number of steps.')

    # Tensorboard arguments.
    parser.add_argument('--logdir', type=str, default='../logs/rnn/logs',
                        help='Tensorboard log directory.')
    parser.add_argument('--log_every', type=int, default=400,
                        help='Log for tensorboard every number of steps.')

    args = parser.parse_args()

    main()
