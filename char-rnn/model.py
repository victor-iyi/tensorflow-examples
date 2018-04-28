from supress import *

from tensorflow.contrib import rnn


class Model:
    """Multi-layer Recurrent Neural Networks (LSTM, RNN) for 
    character-level language models.

    To learn more about character-rnn, 
    Visit Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

    Arguments:
        args {argparse.ArgumentParser} -- Command line arguments from train.py

    Keyword Arguments:
        training {bool} -- [Training mode.] (default: {True})

    Raises:
        ValueError -- Model type not supported. Supported types include:
                            RNN, LSTM, GRU and NAS.
    """

    def __init__(self, args, training=True):
        self.args = args

        # Set batch size & sequence length to 1 if not in traiing mode.
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # Recurrent Architecture.
        if args.model.lower() == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model.lower() == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model.lower() == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model.lower() == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise ValueError("Model type not supported.")

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)

            # Add dropout only during training.
            if training and (args.input_keep_probs < 1.0 or args.output_keep_probs < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob, output_keep_prob=args.output_keep_prob)

            # Append the hidden cell.
            cells.append(cell)

        # Recurrent Cell.
        self.cell = rnn.MultiRNNCell(cell, state_is_tuple=True)

        # Model placeholders
        self.input_data = tf.placeholder(dtype=tf.int32,
                                         shape=[args.batch_size, args.seq_length])
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size,
                                             dtype=tf.int32)

        # Recurrent Neural Net Language Modelling.
        with tf.variable_scope(name='rnnlm'):
            softmax_W = tf.get_variable(name='softmax_W',
                                        shape=[args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable(name='softmax_b',
                                        shape=[args.vocab_size])

        # Embeddings.
        embedding = tf.get_variable('embedding',
                                    shape=[args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # Dropout input embeddings.
        if training:
            inputs = tf.nn.dropout(inputs, keep_prob=args.input_keep_prob)
        
        inputs = tf.split(axis=1, value=inputs, num_split=args.seq_length)
        inputs = [tf.squeeze(input_, axis=[1]) for input_ in inputs]

