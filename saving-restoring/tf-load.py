"""Using pre-trained graph in new graph.
"""
import os

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

saved_dir = os.path.join('saved/saving-restoring/', 'tf-load')
meta_path = os.path.join(saved_dir, 'model.ckpt.meta')
data_path = os.path.join(saved_dir, 'model.ckpt.data-00000-of-00001')

# vgg = tf.train.import_meta_graph(meta_path)

# graph = tf.get_default_graph()

print(tf.VERSION)
