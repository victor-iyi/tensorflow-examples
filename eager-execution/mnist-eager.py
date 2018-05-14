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
import tensorflow as tf


def load_data():
    """Load the MNIST dataset into train & testing set.

    Returns:
        train, test (tuple): Training and testing set.
    """
    train, test = tf.keras.datasets.mnist.load_data()

    return train, test


def main():
    pass


if __name__ == '__main__':
    main()
