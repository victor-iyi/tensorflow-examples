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
            self.args.batch_size = 1
            self.args.seq_length = 1

        # Recurrent Architecture.
        if self.args.model.lower() == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.args.model.lower() == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif self.args.model.lower() == 'gru':
            cell_fn = rnn.GRUCell
        elif self.args.model.lower() == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise ValueError("Model type not supported.")

        cells = []
        for _ in range(self.args.num_layers):
            cell = cell_fn(self.args.rnn_size)

            # Add dropout only during training.
            if training and (self.args.input_keep_probs < 1.0 or self.args.output_keep_probs < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=self.args.input_keep_prob, output_keep_prob=self.args.output_keep_prob)

            # Append the hidden cell.
            cells.append(cell)

        # Recurrent Cell.
        self.cell = rnn.MultiRNNCell(cell, state_is_tuple=True)

        # Model placeholders
        self.input_data = tf.placeholder(dtype=tf.int32,
                                         shape=[self.args.batch_size, self.args.seq_length])
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[self.args.batch_size, self.args.seq_length])
        self.initial_state = cell.zero_state(self.args.batch_size,
                                             dtype=tf.int32)
