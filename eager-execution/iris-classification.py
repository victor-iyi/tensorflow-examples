"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: iris-classification.py
    Created on 06 May, 2018 @ 4:59 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

from sklearn import datasets, preprocessing, model_selection

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

# Turn on eager execution.
tf.enable_eager_execution()

# Load the Iris dataset.
data = datasets.load_iris()
TARGET_NAMES = {i: l for i, l in enumerate(data['target_names'])}

print(55 * '-')
print('Feature names: {feature_names}'.format(**data))
print('Target names: {target_names}'.format(**data))
print('Target mapping: {}'.format(TARGET_NAMES))
print(55 * '-', '\n')

# Pre-processing step.
features = preprocessing.MinMaxScaler(feature_range=(-1, 1))
features = features.fit_transform(data['data'])

labels = preprocessing.OneHotEncoder(sparse=False)
labels = labels.fit_transform(data['target'].reshape(-1, 1))

# Split into training and testing sets.
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels,
                                                                    test_size=0.25)

train_data = tf.data.Dataset.from_tensor_slices((X_train, X_test))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

print(55 * '-')
print('X_train.shape = {}\tX_test.shape:{}'.format(X_train.shape, X_test.shape))
print('y_train.shape = {}\ty_test.shape:{}'.format(y_train.shape, y_test.shape))
print(55 * '-', '\n')

# Model variables
W = tfe.Variable(0.3, name='W')
b = tfe.Variable(0., name='b')


def dense(X, W, b):
    return tf.matmul(X, W) + b


hidden = tf.layers.Dense(units=10, activation=tf.nn.relu, name='hidden_layer')
dropout = tf.layers.Dropout(name='dropout')
output = tf.layers.Dense(units=3, activation=None, name='output')


class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()

        self.hidden_layer = tf.layers.Dense(units=10,
                                            activation=tf.nn.relu,
                                            use_bias=True)
        self.output_layer = tf.layers.Dense(units=3,
                                            activation=None,
                                            use_bias=True)

    def __call__(self, x, **kwargs):
        return self.call(x, **kwargs)

    def call(self, x, **kwargs):
        return self.output_layer(self.hidden_layer(x))


net = Network()

# Get the variable counts before any op.
var_len = len(net.variables)
print('var_len = {}'.format(var_len))

# Make predictions
pred_values = tf.random_normal(shape=[5, 4])
pred = net(pred_values)
print('pred = {}'.format(pred))

# Get the variable counts after making predictions.
var_len = len(net.variables)
print('var_len = {}'.format(var_len))


def loss(model: Network, inputs: any, labels: any):
    logits = model(inputs)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                         labels=labels)
    return tf.reduce_mean(entropy, name="loss")


optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)


def train_step(loss: loss, model: Network, optimizer: tf.train.Optimizer, x, y):
    optimizer.minimize(loss=lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())
