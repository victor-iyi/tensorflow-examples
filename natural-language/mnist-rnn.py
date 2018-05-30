"""Recurrent Neural Network implementation to classify MNIST handwritten digits.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: mnist-rnn.py
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


# noinspection PyAttributeOutsideInit
class RNN:
    def __init__(self, args):
        self.args = args

        # Initialize model's weights.
        with tf.name_scope("weights"):
            self.Wx = tf.get_variable(name="Wx",
                                      shape=(self.args.time_steps, self.args.hidden_size),
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            self.variable_summaries(self.Wx)

            self.Wh = tf.get_variable(name="Wh",
                                      shape=(self.args.hidden_size, self.args.hidden_size),
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            self.variable_summaries(self.Wh)

            self.Wo = tf.get_variable(name="Wo",
                                      shape=(self.args.hidden_size, self.args.num_classes),
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
            self.variable_summaries(self.Wo)

        # Initialize models bias.
        with tf.name_scope("biases"):
            self.bh = tf.get_variable(name="bh",
                                      shape=[self.args.hidden_size],
                                      initializer=tf.zeros_initializer())
            self.variable_summaries(self.bh)

            self.bo = tf.get_variable(name="bo",
                                      shape=[self.args.num_classes],
                                      initializer=tf.zeros_initializer())
            self.variable_summaries(self.bo)

        self.global_step = tf.train.get_global_step()
        # self.summary = tf.summary.merge_all()

    def __call__(self, inputs: tf.Tensor):
        return self.predict(inputs)

    def predict(self, inputs: tf.Tensor):
        # Reshape inputs.
        with tf.name_scope('inputs'):
            inputs = tf.transpose(inputs, perm=[1, 0, 2], name="reshape")

        with tf.name_scope('hidden_states'):
            init_state = tf.zeros(shape=(self.args.batch_size, self.args.hidden_size),
                                  name="initial")
            hidden_states = tf.scan(self._loop, inputs, initializer=init_state, name="states")

        with tf.name_scope('output'):
            outputs = tf.map_fn(self._output, hidden_states, name="output_states")

            predictions = {
                'logits': outputs[-1],
                'probability': tf.nn.softmax(outputs[-1], name="probability")
            }

        return predictions

    def train(self, features: tf.Tensor, labels: tf.Tensor):
        pred = self.predict(features)
        logits = pred['logits']

        with tf.name_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                                        logits=logits,
                                                        reduction=tf.losses.Reduction.MEAN)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss,
                                                    global_step=self.global_step,
                                                    name="train_op")

        # Merge all summaries.
        # self.summary = tf.summary.merge_all()

    def eval(self, features: tf.Tensor, labels: tf.Tensor):
        pred = self.predict(features)
        y_pred = tf.argmax(pred['probability'], axis=1)
        y_true = tf.argmax(labels, axis=1)

        with tf.name_scope('accuracy'):
            correct = tf.equal(y_pred, y_true)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

            tf.summary.scalar('accuracy', accuracy)

        return accuracy

    @staticmethod
    def variable_summaries(var: tf.Tensor):
        with tf.name_scope('summaries'):
            # Mean value.
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # Standard deviation.
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

            # Min & Max values.
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('max', tf.reduce_max(var))

            # Distribution.
            tf.summary.histogram('histogram', var)

    def _loop(self, prev: tf.Tensor, curr: tf.Tensor):
        hidden = tf.matmul(curr, self.Wx) + tf.matmul(prev, self.Wh) + self.bh
        return tf.tanh(hidden)

    def _output(self, hidden: tf.Tensor):
        return tf.matmul(hidden, self.Wo) + self.bo

    @property
    def parameters(self):
        return [self.Wx, self.Wh, self.Wo, self.bh, self.bo]


def main():
    train, test = load_data(one_hot=True, dataset=True)

    iterator = tf.data.Iterator.from_structure(output_types=train.output_types,
                                               output_shapes=train.output_shapes,
                                               output_classes=train.output_classes)
    # Get features & labels.
    features, labels = iterator.get_next()

    # Train & test iterator object.
    train_iter = iterator.make_initializer(train, name="train_dataset")
    # test_iter = iterator.make_initializer(test, name="test_dataset")

    with tf.Session() as sess:
        model = RNN(args=args)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logdir=args.logdir, graph=sess.graph)
        merge = tf.summary.merge_all()

        # Restore Training session properly.
        if tf.gfile.Exists(args.save_dir):
            try:
                ckpt_path = tf.train.latest_checkpoint(args.save_dir)
                saver.restore(sess=sess, save_path=ckpt_path)
                print('INFO: Restored checkpoint from {}'.format(ckpt_path))
            except Exception as e:
                print('WARN: Could not load checkpoint file. {}'.format(e))
                sess.run(init)
        else:
            tf.gfile.MakeDirs(args.save_dir)
            print('INFO: Created checkpoint directory: {}'.format(args.save_dir))
            sess.run(init)

        # Train dataset initializer.
        sess.run(train_iter)

        for epoch in range(args.epochs):
            try:
                model.train(features, labels)
                acc = model.eval(features, labels)

                _, _step, _loss, summary = sess.run([model.train_op, model.global_step,
                                                     model.loss, merge])
                writer.add_summary(summary=summary, global_step=_step)

                if args.save_every % _step == 0:
                    print('\nSaving checkpoint to {}'.format(args.save_dir))
                    saver.save(sess=sess, save_path=args.save_dir,
                               global_step=model.global_step)

                print('Epoch: {:,}\tStep: {:,}\tAcc: {:.2%}\tLoss: {:.3f}'
                      .format(epoch + 1, _step, acc, _loss))
            except KeyboardInterrupt:
                print('\nTraining interrupted by user!')
                print('Saving checkpoint to {}'.format(args.save_dir))
                saver.save(sess=sess, save_path=args.save_dir,
                           global_step=model.global_step)
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training iteration/epochs.')
    parser.add_argument('--save_dir', type=str, default='../saved/mnist-rnn/model',
                        help='Model save directory.')
    parser.add_argument('--save_every', type=int, default=200,
                        help='Save model every number of steps.')

    # Tensorboard arguments.
    parser.add_argument('--logdir', type=str, default='../logs/mnist-rnn/logs',
                        help='Tensorboard log directory.')
    parser.add_argument('--log_every', type=int, default=400,
                        help='Log for tensorboard every number of steps.')

    args = parser.parse_args()

    main()
