"""Suppress TensorFlow's version deprecation warnings."""

# Backward compatibility with Python 2.
from __future__ import print_function

# Suppress all warnings.
import warnings

warnings.filterwarnings('ignore')

# Common imports.
import tensorflow as tf
