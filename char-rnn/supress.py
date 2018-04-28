"""Supress TensorFlow's version deprecation warnings."""

# Backward compactibility with Python 2.
from __future__ import print_function

# Supress all warnings.
import warnings
warnings.filterwarnings('ignore')

# Common imports.
import tensorflow as tf
