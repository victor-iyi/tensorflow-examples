"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: eager-exec.py
    Created on 03 May, 2018 @ 2:11 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable eager execution.
tf.enable_eager_execution()

train_data_url = "http://download.tensorflow.org/data/iris_training.csv"

train_data_fp = tf.keras.utils.get_file(fname=os.path.basename(train_data_url), origin=train_data_url)

print("Local copy of iris dataset in {}".format(train_data_fp))


def parse_csv(line):
    # Sets the dtype.
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)

    # Features - 1st 4 columns.
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Labels - last column.
    label = tf.reshape(parsed_line[-1], shape=())

    return features, label
