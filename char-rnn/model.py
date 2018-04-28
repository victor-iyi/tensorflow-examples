from supress import *

from tensorflow.contrib import rnn


class Model:

    def __init__(self, args, training=True):
        self.args = args

        # Set batch size & sequence length to 1 if not in traiing mode.
        if not training:
            self.args.batch_size = 1
            self.args.seq_length = 1

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

