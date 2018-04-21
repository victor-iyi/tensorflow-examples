"""TensorFlow has two major kinds of "shape"
- The Static Shape
- The Dynamic Shape
"""

# To ignore tensorflow version warning
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
print(tf.VERSION)

# Demo: TF Static Shape
my_tensor = tf.ones(shape=[8, 2])
print('my_tensor = {}'.format(my_tensor))

# Retrieve it's static shape (NOTE: Static ops are attached to TF Tensor
# & usually have underscores in their names.
static_shape = my_tensor.get_shape()
print('static_shape = {}'.format(static_shape))

print('static_shape.as_list() = {}'.format(static_shape.as_list()))

# Create a placeholder with undefined shape.
my_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 2])
print('my_placeholder = {}'.format(my_placeholder))

# Update the shape.
print('BEFORE: my_placeholder.get_shape() = {}'.format(
    my_placeholder.get_shape()))

my_placeholder.set_shape([8, 2])
print('AFTER: my_placeholder.get_shape() = {}'.format(
    my_placeholder.get_shape()))

# Line divider.
print('\n\n', 70 * '=', '\n\n')

# Demo: TF Dynamic Shape
my_tensor = tf.ones(shape=[8, 2])
print('my_tensor = {}'.format(my_tensor))

# Retrieve it's dynamic shape (NOTE: Dynamic ops are attached to d main scope
# & usually have no underscores in their names.
my_dynamic_shape = tf.shape(my_tensor)
print('my_dynamic_shape = {}'.format(my_dynamic_shape))

# Dynamic shape is a tensor itself describing the shape of the original
# tensor.
my_tensor_reshaped = tf.reshape(tensor=my_tensor, shape=[2, 4, 2])
print('my_tensor_reshaped = {}'.format(my_tensor_reshaped))

# To access the dynamic shape's value, you need to run it through a Session
dynamic_value = my_dynamic_shape.eval(session=tf.Session())
print(dynamic_value)
