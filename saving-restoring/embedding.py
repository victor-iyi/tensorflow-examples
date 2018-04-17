import os
import logging

# Ignore TensorFlow's deprecation warnings.
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

# Configure log level.
logging.basicConfig(level=logging.DEBUG)

# Dataset directory.
data_path = os.path.join('datasets/', 'python_code.py')

# Save model & Tensorboard log dir.
save_dir = os.path.join('saved/saving-restoring', 'embedding')

# Tensorboard event directory.
tensorboard_dir = os.path.join(save_dir, 'tensorboard')
logdir = os.path.join(tensorboard_dir, 'log')

# Model saver directory.
model_dir = os.path.join(save_dir, 'models')
model_path = os.path.join(model_dir, 'model.ckpt')


def batch_generator(corpus, batch_size, seq_length):
    """Batch generator function for generating the next batch
    of a given corpus during training.

    Arguments:
        corpus {str} -- Data corpus.
        batch_size {int} -- Size of the generated batch.
        seq_length {int} -- Length of a single sequence in the dataset.

    Raises:
        ValueError -- seq_len >= len(corpus)

    Yeilds:
        x, y {list} -- Next training batch with it's corresponding
        label.
    """

    if seq_length >= len(corpus):
        raise ValueError('seq_length >= len(corpus): {}>={}'.format(
            seq_length, len(corpus)))

    sequences = [corpus[i: i + seq_length]
                 for i in range(len(corpus) - seq_length)]
    ys = [corpus[i: i + 1] for i in range(seq_length, len(corpus))]

    for i in range(0, len(sequences), batch_size):
        x = sequences[i: i + batch_size]
        y = ys[i: i + batch_size]

        yield x, y


# Load the corpus.
with open(data_path, mode='r', encoding='utf-8') as f:
    text = f.read()

corpus = text.split()
corpus_len = len(corpus)

# Corpus unique tokens.
tokens = set(corpus)
nb_tokens = len(tokens)

# Mapping between word to id & vice versa.
word2id = {w: i for i, w in enumerate(tokens)}
id2word = {i: w for i, w in enumerate(tokens)}

# Convert the corpus to ids.
corpus_ids = [word2id[word] for word in corpus]

# Log corpus & token count.
logging.info('corpus_len = {:,}'.format(corpus_len))
logging.info('nb_tokens  = {:,}'.format(nb_tokens))

# Model placeholders (inputs).
seq_len = 5
with tf.name_scope("placeholders"):
    X = tf.placeholder(dtype=tf.int32, shape=[None, seq_len], name="X")
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="Y")

# Word Embeddings.
embedding_dim = 30
with tf.name_scope("embedding"):
    embedding = tf.get_variable("embedding", dtype=tf.float32,
                                shape=[nb_tokens, embedding_dim],
                                initializer=tf.truncated_normal_initializer)
    # Context vector. [batch_size x seq_len x embedding_dim]
    context_vec = tf.nn.embedding_lookup(embedding, X, name="lookup")
    context_vec = tf.reshape(context_vec,
                             shape=[tf.shape(X)[0], seq_len*embedding_dim])

# BUILD THE MODEL.
with tf.name_scope("layer1"):
    W1 = tf.get_variable("weights", dtype=tf.float32,
                         shape=[seq_len*embedding_dim, embedding_dim],
                         initializer=tf.truncated_normal_initializer)
    b1 = tf.get_variable("biases", dtype=tf.float32,
                         shape=[embedding_dim],
                         initializer=tf.zeros_initializer)
    # First hidden layer.
    h1 = tf.nn.relu(tf.matmul(context_vec, W1) + b1)

with tf.name_scope("layer2"):
    W2 = tf.get_variable("weights", dtype=tf.float32,
                         shape=[embedding_dim, embedding_dim],
                         initializer=tf.truncated_normal_initializer)
    b2 = tf.get_variable("biases", dtype=tf.float32,
                         initializer=tf.zeros_initializer)

    # Second hidden layer.
