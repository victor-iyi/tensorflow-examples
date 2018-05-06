"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: intro.py
    Created on 06 May, 2018 @ 4:12 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python import tfe
from sklearn import datasets, preprocessing, model_selection

# Enable eager mode.
# tf.enable_eager_execution()

data = datasets.load_iris()

TARGET_NAMES = {i: l for i, l in enumerate(data['target_names'])}

features = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(data['data'])
labels = preprocessing.OneHotEncoder(sparse=False).fit_transform(data['target'].reshape(-1, 1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.1)


def square_func(W):
    return tf.square(W)


f_grad = tfe.gradients_function(square_func, params=['W'])
print(f_grad(tf.constant(0.3)))
