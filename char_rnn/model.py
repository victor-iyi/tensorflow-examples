from .suppress import *

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model:
    """Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models.

    To learn more about character-rnn,
    Visit Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

    Arguments:
        args {argparse.ArgumentParser} -- Command line arguments from train.py

    Keyword Arguments:
        training {bool} -- Training mode. (default: {True})

    Raises:
        ValueError -- Model type not supported. Supported types include:
                            RNN, LSTM, GRU and NAS.
    """

    def __init__(self, args, training=True):
        self.args = args

        # Set batch size & sequence length to 1 if not in training mode.
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

        # Construct the hidden layers' cell.
        cell = None
        cells = []

        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)

            # Add dropout only during training.
            if training and (args.input_keep_prob < 1.0 or args.output_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)

            # Append the hidden cell.
            cells.append(cell)

        # Recurrent Cell.
        self.cell = rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

        # Model placeholders.
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length], name="input_data")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length], name="targets")
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.int32)

        # Recurrent Neural Net Language Modelling.
        with tf.variable_scope('rnnlm'):
            softmax_W = tf.get_variable(name='softmax_W', shape=[args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable(name='softmax_b', shape=[args.vocab_size])

        # Embeddings.
        embedding = tf.get_variable('embedding', shape=[args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # Dropout input embeddings.
        if training:
            inputs = tf.nn.dropout(inputs, keep_prob=args.input_keep_prob)

        # Split & reshape inputs.
        inputs = tf.split(value=inputs, num_or_size_splits=args.seq_length, axis=1)
        inputs = [tf.squeeze(input_, axis=[1]) for input_ in inputs]

        def loop(prev, _):
            """Function to be performed at each recurrent layer.

            This function will be applied to the i-th output in order to generate the i+1-st input, and
            decoder_inputs will be ignored, except for the first element ("GO" symbol). This can be used
             for decoding, but also for training to  emulate http://arxiv.org/abs/1506.03099.

            Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

            Arguments:
                prev {tf.Tensor} -- prev is a 2D Tensor of shape [batch_size x output_size].
                _ {tf.Tensor} -- i is an integer, the step number (when advanced control is needed).

            Returns:
                {tf.Tensor} -- A 2D Tensor of shape [batch_size, input_size] which represents
                the embedding matrix of the predicted next character.
            """
            prev = tf.matmul(prev, softmax_W) + softmax_b
            prev_symbol = tf.stop_gradient(input=tf.arg_max(prev, dimension=1))
            return tf.embedding_lookup(embedding, prev_symbol)

        # Decoder.
        outputs, prev_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if not training else None,
                                                         scope='rnnlm')

        outputs = tf.reshape(tf.concat(outputs, axis=1), shape=[-1, args.rnn_size])

        # Fully connected & softmax layer.
        self.logits = tf.matmul(outputs, softmax_W) + softmax_b
        self.probs = tf.nn.softmax(self.logits, name="probs")

        # Loss function.
        with tf.variable_scope('loss'):
            seq_loss = legacy_seq2seq.sequence_loss_by_example(
                logits=self.logits,
                targets=tf.reshape(self.targets, shape=[-1]),
                weights=[tf.ones(shape=[args.batch_size * args.seq_length])])

            self.loss = tf.reduce_sum(seq_loss) / args.batch_size / args.seq_length

        self.final_state = prev_state

        self.lr = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Trainable variables & gradient clipping.
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(self.loss, tvars),
                                          clip_norm=args.grad_clip)

        # Optimizer.
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

        # Train ops.
        self.train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars),
                                                  global_step=self.global_step,
                                                  name="train_op")

        # Tensorboard.
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('seq_loss', seq_loss)
        tf.summary.scalar('loss', self.loss)

    def sample(self, sess: tf.Session, chars: tuple, vocab: dict,
               num: int = 200, prime: str = 'The', sampling_type: int = 1):
        """Sample from the prediction probability one character at a time.

        Arguments:
            sess {tf.Session} -- Session containing the default graph.
            chars {tuple} -- List of characters in the vocab.
            vocab {dict} -- Mapping from character to id. Dictionary containing characters & corresponding numeric
            value.

        Keyword Arguments: num {int} -- Number of character to predict. (default: {200}) prime {str} -- Beginning of
        prediction sequence. (default: {'The'}) sampling_type {int} -- Description of how to choose the top most
        likely character. Options are 1, 2, & 3. (default: {1})

         Returns:
             ret {str} -- Sequence containing the prediction of the `num` characters.
        """

        # Initial cell state. TODO: Change dtype=tf.float32
        # Predict final state given input data & prev state.
        state = sess.run(self.cell.zero_state(batch_size=1, dtype=tf.int32))
        for char in prime[:-1]:
            # Input data: one char at a time.
            x = np.zeros(shape=(1, 1))
            x[0, 0] = vocab[char]

            # Given input data & initial state, predict final state.
            feed_dict = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        def weighted_pick(weights):
            c = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(c, np.random.rand(1) * s))

        # Initial character.
        ret = prime
        char = prime[-1]

        # Prediction loop.
        for i in range(num):
            x = np.zeros(shape=(1, 1))
            x[0, 0] = vocab[char]

            # Predict probability of next word & prev state.
            feed_dict = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed_dict=feed_dict)

            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # Sampling type = 1 (default)
                sample = weighted_pick(p)

            # Get the character representation of sampled character.
            pred = chars[sample]
            ret += pred
            char = pred

        return ret
