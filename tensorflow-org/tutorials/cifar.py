"""Classify cifar10 dataset using tf.estimator API.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: cifar.py
     Created on 20 May, 2018 @ 4:51 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import argparse

import numpy as np
import tensorflow as tf

# Rest TensorFlow's default graph.
tf.reset_default_graph()

# TensorFlow log level (see what's going on during training).
tf.logging.set_verbosity(tf.logging.INFO)

# Command line argument.
args = None


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


def load_data(one_hot: bool = False):
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
    # Download dataset.
    train, test = tf.keras.datasets.cifar10.load_data()

    # Split into features & labels.
    X_train, y_train = train
    X_test, y_test = test

    # Pre-process the images.
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    # Convert to one-hot.
    if one_hot:
        y_train = make_one_hot(indices=y_train, depth=args.num_classes)
        y_test = make_one_hot(indices=y_test, depth=args.num_classes)

    return (X_train, y_train), (X_test, y_test)


def make_dataset(features: np.ndarray, labels: np.ndarray = None):
    """Create dataset object from features &/or labels.

        Args:
            features (np.ndarray): Feature column.
            labels (np.ndarray): Dataset labels.

        Returns:
            tf.data.Dataset: Pre-processed dataset object.
    """
    features = {args.feature_col: features}

    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)

    # Transform dataset.
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(batch_size=args.batch_size)

    return dataset


def input_fn(features: np.ndarray, labels: np.ndarray = None,
             epochs: int = 1, shuffle: bool = False):
    """Creates input function given features & (maybe) labels.

        Args:
            features (np.ndarray): Input images.
            labels (np.ndarray): Data targets.
            epochs (int): Number of passes through data.
            shuffle (bool): Maybe shuffle dataset.

        Returns:
            Function, that has signature of ()->(dict of `features`, `targets`)
    """
    return tf.estimator.inputs.numpy_input_fn(
        x={args.feature_col: features},
        y=labels,
        batch_size=args.batch_size,
        num_epochs=epochs,
        shuffle=shuffle,
        num_threads=2 if shuffle else 1
    )


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode=tf.estimator.ModeKeys):
    """Construct a 2-layer convolutional network.

        Arguments:
            features (tf.Tensor):
                Dataset images with shape (batch_size, img_flat) or
                (batch_size, img_width, img_height, img_depth).

            labels (tf.Tensor):
                Dataset labels (one-hot encoded).

            mode (tf.estimator.ModeKeys):
                One of tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.TRAIN,
                or tf.estimator.ModeKeys.EVAL.

        Returns:
            tf.estimator.EstimatorSpec:
                Ops and objects returned from `model_fn` and passed to
                tf.estimator.Estimator.
    """
    with tf.name_scope("model"):
        # Network layers.
        with tf.name_scope("layers"):
            # Input Layer
            with tf.name_scope("input"):
                input_layer = tf.reshape(tensor=features[args.feature_col],
                                         shape=(-1, args.img_size, args.img_size, args.img_depth),
                                         name="reshape")

            # 1st convolutional block.
            with tf.name_scope("conv_block_1"):
                conv1 = tf.layers.conv2d(inputs=input_layer,
                                         filters=args.filter_conv1,
                                         kernel_size=args.kernel_size,
                                         activation=tf.nn.relu, padding="same")

                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=args.pool_size,
                                                strides=2, name="pooling")

            with tf.name_scope("conv_block_2"):
                conv2 = tf.layers.conv2d(inputs=pool1,
                                         filters=args.filter_conv2,
                                         kernel_size=args.kernel_size,
                                         activation=tf.nn.relu, padding="same")

                pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                pool_size=args.pool_size,
                                                strides=2, name="pooling")

            with tf.name_scope("fully_connected"):
                flatten = tf.layers.flatten(inputs=pool2, name="flatten")

                # Fully connected layer.
                dense = tf.layers.dense(inputs=flatten, units=args.dense_units,
                                        activation=tf.nn.relu, name="dense")

                # Dropout for regularization (to prevent network from overfitting).
                dropout = tf.layers.dropout(inputs=dense,
                                            rate=args.dropout,
                                            training=mode == tf.estimator.ModeKeys.TRAIN,
                                            name="dropout")

            with tf.name_scope("output"):
                logits = tf.layers.dense(inputs=dropout, units=args.num_classes,
                                         name="logits")

        # Predictions.
        with tf.name_scope("prediction"):
            predictions = {
                "classes": tf.argmax(input=logits, axis=1, name="classes"),
                "probabilities": tf.nn.softmax(logits=logits, name="probabilities"),
            }

        # Return predictions (if mode == PREDICT).
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Estimate the loss.
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits,
                                               reduction=tf.losses.Reduction.MEAN)

        # Train the model.
        with tf.name_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate,
                                                  decay=args.decay_rate)
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_or_create_global_step(),
                                          name="train_op")

        # Evaluate accuracy.
        with tf.name_scope("evaluate"):
            # Evaluation metrics operation.
            eval_metrics_op = {
                "accuracy": tf.metrics.accuracy(labels=labels,
                                                predictions=predictions["probabilities"],
                                                name="accuracy")
            }

    # Return training operation (if mode == TRAIN).
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          loss=loss, train_op=train_op,
                                          # Optional: Accuracy during training.
                                          eval_metric_ops=eval_metrics_op)

    # Return evaluation metrics (if mode == EVAL).
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metrics_op)


def main():
    train, test = load_data(one_hot=True)

    X_train, y_train = train
    X_test, y_test = test

    log_tensors = {
        # "labels": "fifo_queue_DequeueUpTo:2",
        "output": "model/prediction/classes:0",
    }
    hooks = tf.train.LoggingTensorHook(tensors=log_tensors,
                                       every_n_iter=args.log_every,
                                       at_end=True)

    # Classifier.
    clf = tf.estimator.Estimator(model_fn=model_fn,
                                 model_dir=args.logdir)

    # Train the model.
    train_input_fn = input_fn(features=X_train, labels=y_train,
                              epochs=args.epochs, shuffle=True)
    clf.train(train_input_fn, hooks=[hooks], max_steps=args.steps)

    # Evaluate the model.
    eval_input_fn = input_fn(features=X_test, labels=y_test, epochs=1)
    results = clf.evaluate(input_fn=eval_input_fn)

    print('Global steps = {:,}\tAccuracy = {:.02%}\tLoss = {:.4f}'
          .format(results['global_step'], results['accuracy'], results['loss']))


if __name__ == '__main__':
    # Command line argument parser.
    parser = argparse.ArgumentParser()

    # Input arguments.
    parser.add_argument('--img_size', type=int, default=32,
                        help="Image size. The default for CIFAR10 data is 32")
    parser.add_argument('--img_depth', type=int, default=3,
                        help="Image channel. The default for CIFAR10 data is 3, "
                             "which signifies image is a colored image.")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of classes to be predicted.")

    # Dataset arguments.
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Mini batch size. Use lower batch size if running on CPU.")
    parser.add_argument('--shuffle_rate', type=int, default=1000,
                        help="Dataset shuffle rate. A fixed size buffer from which the "
                             "next element will be uniformly chosen from.")
    parser.add_argument('--data_transform_count', type=int, default=5,
                        help="Dataset transform repeat count. "
                             "Use smaller (or 1) if running on CPU")
    parser.add_argument('--feature_col', type=str, default="images",
                        help="Feature column label for tf.feature_column")

    # Estimator arguments.
    parser.add_argument('--save_dir', type=str, default="../../saved/tutorials/cifar",
                        help="Specifies the directory where model data "
                             "(checkpoints) will be saved.")
    parser.add_argument('--logdir', type=str, default="../../logs/tutorials/cifar",
                        help="Specifies the directory where model data "
                             "(checkpoints) will be saved.")
    parser.add_argument('--log_every', type=int, default=50,
                        help="Log specified tensors every ``log_every`` iterations.")

    # Network arguments.
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Kernel size for each convolution. "
                             "default [5, 5]")
    parser.add_argument('--pool_size', type=int, default=2,
                        help="Down-sampling filter size. default [2, 2]")
    parser.add_argument('--filter_conv1', type=int, default=32,
                        help="Size of 1st convolutional filters.")
    parser.add_argument('--filter_conv2', type=int, default=64,
                        help="Size of 2nd convolutional filters.")
    parser.add_argument('--dense_units', type=int, default=1024,
                        help="Number of neurons in the fully connected layer.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout regularization rate (probability that a given "
                             "element will be dropped during training).")

    # Training & optimizer arguments.
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs. Signifies the number of "
                             "times to loop through a complete training iteration. "
                             "Default is `None` meaning that the  model will train "
                             "until the specified number of steps is reached.")
    parser.add_argument('--steps', type=int, default=1000,
                        help="Number of training steps. Represents the number of "
                             "times to loop through a complete mini-batch cycle.")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate for RMSPropOptimizer.")
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help="Decay rate for RMSPropOptimizer.")

    # Parse command line arguments.
    args = parser.parse_args()
    print(args)

    # Start program execution.
    main()
