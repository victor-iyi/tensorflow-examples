import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import batch_and_drop_remainder

# Command line arguments.
args = None


def make_one_hot(indices: np.ndarray, depth: int, dtype: np.dtype = np.int32):
    hot = np.zeros(shape=[len(indices), depth], dtype=dtype)

    for i, index in enumerate(indices):
        hot[i:, index] = 1.

    return hot


def make_dataset(features: np.ndarray, labels: np.ndarray = None, shuffle: bool = False):
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


def main():
    train, test = load_data(one_hot=True, dataset=True)
    iterator = tf.data.Iterator.from_structure(output_types=train.output_types,
                                               output_shapes=train.output_shapes,
                                               output_classes=train.output_classes)
    features, labels = iterator.get_next()

    # RNN Cell
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=args.hidden_size)
    initial_state = cell.zero_state(batch_size=args.batch_size,
                                    dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=features,
                                        initial_state=initial_state,
                                        dtype=tf.float32)
    # Output at last time step.
    rnn_output = outputs[:, -1]

    # Fully connected layer.
    logits = tf.layers.dense(inputs=rnn_output, units=args.num_classes,
                             name="logits")
    y_pred = tf.nn.softmax(logits, name="probabilities")

    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits,
                                               reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss=loss,
                                      global_step=global_step,
                                      name='train_op')
        tf.summary.scalar('global_step', global_step)

    with tf.name_scope('evaluation'):
        y_true = tf.argmax(labels, axis=1)
        y_pred_true = tf.argmax(y_pred, axis=1)

        correct = tf.equal(y_pred_true, y_true, name='correct')
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),
                                  name='accuracy')

        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        save_path = os.path.join(args.save_dir, 'model.ckpt')

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logdir=args.logdir, graph=sess.graph)

        init = tf.global_variables_initializer()

        if tf.gfile.Exists(args.save_dir):
            try:
                ckpt_path = tf.train.latest_checkpoint(args.save_dir)
                saver.restore(sess=sess, save_path=ckpt_path)
                print('INFO: Restored checkpoint from {}'.format(ckpt_path))
            except Exception as e:
                print('WARN: Could not restore checkpoint. {}'.format(e))
                sess.run(init)
        else:
            tf.gfile.MakeDirs(args.save_dir)
            print('INFO: Creating checkpoint files: {}'.format(args.save_dir))
            sess.run(init)

        # Train & test dataset iterator.
        train_iter = iterator.make_initializer(train, name="train_iter")
        # test_iter = iterator.make_initializer(test, name="test_iter")

        for epoch in range(args.epochs):
            try:
                # Initialize train iterator.
                sess.run(train_iter)
                while True:
                    try:
                        _, _step, _loss, _acc = sess.run([train_op, global_step,
                                                          loss, accuracy])

                        print('\rEpoch: {:,}\tStep: {:,}\tAcc: {:.2%}\tLoss: {:.3f}'
                              .format(epoch + 1, _step, _acc, _loss), end='')

                        if _step % args.log_every == 0:
                            summary = sess.run(merged)
                            writer.add_summary(summary, global_step=_step)

                        if _step % args.save_every == 0:
                            print('\n{}'.format('-' * 65))
                            print('\nSaving model to {}'.format(save_path))
                            saver.save(sess=sess, save_path=save_path,
                                       global_step=global_step)
                            print('{}\n'.format('-' * 65))

                    except tf.errors.OutOfRangeError:
                        # End batch loop.
                        break
            except KeyboardInterrupt:
                print('\nTraining interrupted by user.')
                print('\n{}'.format('-' * 65))
                print('Saving model to {}'.format(save_path))
                saver.save(sess=sess, save_path=save_path,
                           global_step=global_step)
                print('{}\n'.format('-' * 65))
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini batch size.')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='Shuffle rate.')

    # Training arguments.
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Optimizer\'s learning rate.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training iteration/epochs.')
    parser.add_argument('--save_dir', type=str, default='../saved/demo/',
                        help='Model save directory.')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save model every number of steps.')

    # Tensorboard arguments.
    parser.add_argument('--logdir', type=str, default='../logs/demo/',
                        help='Tensorboard log directory.')
    parser.add_argument('--log_every', type=int, default=200,
                        help='Log for tensorboard every number of steps.')

    args = parser.parse_args()

    main()
