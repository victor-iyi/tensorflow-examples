"""MNIST classification using TensorFlow's Eager execution mode.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: mnist-eager.py
     Created on 14 May, 2018 @ 5:49 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.

"""
import warnings

# Ignore TensorFlow's deprecation warnings. (ANNOYING!)
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

# Turn on Eager execution mode.
tf.enable_eager_execution()


def load_data():
    """Load the MNIST dataset into train & testing set.

    Returns:
        train, test (tuple): Training and testing set.
    """
    train, test = tf.keras.datasets.mnist.load_data()

    return train, test


def pre_process(features, labels):
    """Flatten images & one-hot encode labels.

    Arguments:
        features {tf.Tensor} -- Dataset images.
        labels {tf.Tensor} -- Dataset labels.

    Returns:
        {(tf.Tensor, tf.Tensor)} -- features, labels
    """

    # Reshaping image to fit the model.
    features = np.array(features, dtype=np.float32)
    img_size_flat = np.prod(features.shape[1:])
    features = features.reshape((-1, img_size_flat))

    # One-hot encoding.
    num_classes = len(np.unique(labels))
    labels = tf.one_hot(indices=labels, depth=num_classes)

    # Return processed features & labels.
    return features, labels


def process_data(features: tf.Tensor, labels: tf.Tensor,
                 batch_size: tf.int32 = 64, buffer_size: tf.int32 = 1000):
    """Create TensorFlow data object from tensor slices.

    Args:
        features (tf.Tensor): Dataset input images.
        labels (tf.Tensor): Dataset one-hot labels.
        batch_size (tf.int32): Mini batch size.
        buffer_size (tf.int32): Buffer size for shuffling the dataset.

    Returns:
        dataset (tf.data.Dataset): TensorFlow's dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset


# noinspection PyAbstractClass
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # self.hidden = tf.keras.layers.Conv2D(filters=5,
        #                                      kernel_size=2,
        #                                      activation='relu')
        # self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        # self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=512)
        self.fc2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, **kwargs):
        # # Conv & Pooling layer
        # result = self.pool(self.hidden(inputs))
        # # Flatten layer.
        # result = self.flatten(result)
        result = inputs
        # Fully connected layers.
        result = self.fc2(self.fc1(result))
        # Output prediction.
        return result


def loss_func(model: tf.keras.Model, features: tf.Tensor, labels: tf.Tensor):
    logits = model(features)
    entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    return tf.reduce_mean(entropy, name="loss")


def compute_grads(model: tf.keras.Model, features: tf.Tensor, labels: tf.Tensor):
    with tfe.GradientTape() as tape:
        loss = loss_func(model=model, features=features, labels=labels)

    return tape.gradient(loss, model.variables)


def main():
    # Logging split.
    print('\n{}'.format(60 * '-'))

    # Load data & split into training & testing sets.
    train, test = load_data()
    X_train, y_train = train
    X_test, y_test = test

    # Number of training/testing samples.
    n_train, n_test = y_train.shape[0], y_test.shape[0]
    print('{:,} train samples\t&\t{:,} testing samples'
          .format(n_train, n_test))

    # Image dimensions.
    img_shape = X_train.shape[1:]
    img_size, img_depth = img_shape[0], 1
    img_size_flat = img_size * img_size * img_depth
    print("Image  = Shape: {}\tSize: {}\tDepth: {}\tFlat: {}"
          .format(img_shape, img_size, img_depth, img_size_flat))

    # Output dimensions.
    classes = np.unique(y_train)
    num_classes = len(classes)
    print('Labels = Classes: {}\tLength: {}'.format(classes, num_classes))

    # Logging split.
    print('{}\n'.format(60 * '-'))

    X_train, y_train = pre_process(X_train, y_train)
    # X_test, y_test = pre_process(X_test, y_test)

    data_train = process_data(X_train, y_train,
                              batch_size=128, buffer_size=1000)
    # data_test = process_data(X_test, y_test,
    #                          batch_size=68, buffer_size=1000)

    epochs = 5
    save_path = './saved/mnist-eager/model'
    save_step = 500

    learning_rate = 1e-2

    model = Model()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.train.get_or_create_global_step()
    saver = tfe.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)

    print('{0}\n\t\tTRAINING STARTED!\n{0}\n'.format(55 * '-'))

    for epoch in range(epochs):
        try:
            for batch, (features, labels) in enumerate(data_train):
                # Calculate the derivative of loss w.r.t. model variables.
                grads = compute_grads(model, features, labels)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=tf.train.get_or_create_global_step())

                loss = loss_func(model=model, features=features, labels=labels)

                # Log training progress.
                print(('\rEpoch: {:,}\tStep: {:,}\tBatch: {:,}'
                       '\tLoss: {:.3f}').format(epoch + 1, global_step.numpy(), batch + 1, loss.numpy()),
                      end='')

                if global_step.numpy() % save_step == 0:
                    print('\nSaving model to {}'.format(save_path))
                    saver.save(save_path)

        except KeyboardInterrupt:
            print('\n{}\nTraining interrupted by user'.format(55 * ''))
            saver.save(file_prefix=save_path)
            print('Model saved to {}'.format(save_path))
            break

    # !- End epochs.
    print('\n\n{0}\n\t\tTRAINING ENDED!\n{0}\n'.format(55 * '-'))


if __name__ == '__main__':
    main()
