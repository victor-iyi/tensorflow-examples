"""A demo of eager execution in TensorFlow.

  @author
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola

  @project
    File: eager-execution.py
    Created on 03 May, 2018 @ 6:17 PM.

  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from iris_data import load_data, SPECIES

# Turn on eager execution.
tf.enable_eager_execution()
print("Eager execution status: {}".format(tf.executing_eagerly()))


# noinspection PyAbstractClass
class Model(tf.keras.Model):
    def __init__(self, n_classes):
        super(Model, self).__init__()

        # Model structure.
        self.hidden1 = tf.keras.layers.Dense(units=10, activation="relu")
        self.hidden2 = tf.keras.layers.Dense(units=10, activation="relu")
        self.prediction = tf.keras.layers.Dense(units=n_classes)

    def call(self, inputs, **kwargs):
        """Implementation of the model's forward pass.
        
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        
        Returns:
            logits {tf.Tensor} -- Un-normalized output.
        """

        logits = self.hidden1(inputs)
        logits = self.hidden2(logits)
        logits = self.prediction(logits)

        return logits


def loss_func(model, x, y):
    """Loss function. How bad is the model doing on the entire training set.

    Args:
        model (tf.keras.Model): Keras model.
        x (dict, tf.data.Dataset): Feature tensor.
        y (tuple, list, tf.data.Dataset): Target tensor.

    Returns:
        loss (tf.Tensor): A real value denoting the loss value of this model.
    """
    y_hat = model(x)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat)
    return tf.reduce_mean(loss, name="loss")


def grad_func(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor):
    """Estimates the derivative of the loss with respect to the model parameters.

    Returns the derivative of the loss w.r.t. weights and biases. The passing
    this to `optimizer.apply_gradients()`` completes the process of apply
    gradient descent.

    Args:
        model (tf.keras.Model): Keras model.
        inputs (tf.Tensor): Input features.
        targets (tf.Tensor): Target outputs.

    Returns:
        gradients (tfe.GradientTape): Gradients of the model w.r.t it's variables.
    """
    with tfe.GradientTape() as tape:
        loss_value = loss_func(model, inputs, targets)

    # Gradient of model w.r.t. it's variables.
    return tape.gradient(loss_value, model.variables)


def process(features: tf.Tensor, labels: tf.Tensor):
    labels = tf.one_hot(labels, len(SPECIES))
    return features, labels


def main():
    # Data dimensions & Hyperparameters.
    n_classes = len(SPECIES)
    learning_rate, epochs = 1e-2, 5
    batch_size, buffer_size = 32, 1000

    # Download and load the dataset.
    train, _ = load_data()

    # Create a TensorFlow Dataset object.
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.map(process)
    train_data = train_data.batch(batch_size=batch_size)
    train_data = train_data.shuffle(buffer_size=buffer_size)

    # Create model object & optimizer.
    model = Model(n_classes=n_classes)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    global_step = tf.train.get_or_create_global_step()

    for epoch in range(1, epochs + 1):
        try:
            # Make an iterator to access the data in the dataset.
            for batch, (features, labels) in enumerate(tfe.Iterator(train_data)):
                # Calculate gradients of loss w.r.t model variables.
                grads = grad_func(model=model, inputs=features, targets=labels)
                # noinspection PyTypeChecker
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=global_step)

                loss = loss_func(model, features, labels)
                print('\rEpoch: {}\tBatch: {:,}\tStep: {:,}\tLoss: {:.3f}'
                      .format(epoch, batch + 1, global_step.numpy(), loss))
        except KeyboardInterrupt:
            print('\nTraining interrupted by user.')
            break


if __name__ == '__main__':
    main()
