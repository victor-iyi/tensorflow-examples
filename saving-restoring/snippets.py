"""Using a pre-trained graph in a new graph

Now that you know how to save and load, you can probably figure out how to do it. Yet, their might be some tricks
that could help you go faster.

    Can the output of one graph be the input of an other graph ?

Yes, but there is a drawback to this: I don’t know yet a way to make the gradient flow easily between graphs,
as you will have to evaluate the first graph, get the results and feed it to the next graph.

This can be ok, until you need to retrain the first graph too. in that case you will need to grab the inputs
gradients to feed it to the training step of your first graph…

    Can I mix all of those different graph in only one graph?

Yes, but you must be careful with namespace. The good point, is that this method simplifies everything: you can load
a pertained VGG-16 for example, access any nodes in the graph, plug your own operations and train the whole thing!

If you only want to fine-tune your own nodes, you can stop the gradients anywhere you want, to avoid training the
whole graph.
"""

import tensorflow as tf
import os.path

# Load the VGG-16 model in the default graph
vgg_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'saved/vgg/vgg-16.meta'))

# Access the graph
vgg_graph = tf.get_default_graph()

# Retrieve VGG inputs
x_plh = vgg_graph.get_tensor_by_name('input:0')

# Choose which node you want to connect your own graph
output_conv = vgg_graph.get_tensor_by_name('conv1_2:0')
# output_conv =vgg_graph.get_tensor_by_name('conv2_2:0')
# output_conv =vgg_graph.get_tensor_by_name('conv3_3:0')
# output_conv =vgg_graph.get_tensor_by_name('conv4_3:0')
# output_conv =vgg_graph.get_tensor_by_name('conv5_3:0')

# Stop the gradient for fine-tuning
# You can also see this as using the VGG model as a feature extractor only
output_conv_sg = tf.stop_gradient(output_conv)  # It's an identity function

# Build further operations
output_conv_shape = output_conv_sg.get_shape().as_list()

W1 = tf.get_variable('W1', shape=[1, 1, output_conv_shape[3], 32], initializer=tf.truncated_normal_initializer)
b1 = tf.get_variable('b1', shape=[32], initializer=tf.zeros_initializer)

z1 = tf.nn.conv2d(output_conv_sg, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
a = tf.nn.relu(z1)
