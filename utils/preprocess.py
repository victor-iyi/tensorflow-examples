import os
import codecs
import collections
import pickle

import numpy as np


class TextLoader:
    """Data loader for character or text dataset.

    Arguments:
        data_dir {str} -- Directory containing input.txt
        batch_size {int} -- Mini batch size.
        seq_length {int} -- Sequence length.

    Keyword Arguments:
        encoding {str} -- Text encoding for reading and writing to files. (default: {'utf-8'})
    """

    def __init__(self, data_dir: str, batch_size: int, seq_length: int, encoding='utf-8'):
        # Arguments and Keyword arguments.
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        # Initialize instance variables to prevent warning.
        self.chars = []
        self.vocab = {}
        self.vocab_size = 0
        self.tensor = None
        self.x_batch = None
        self.y_batch = None
        self.num_batches = 0

        # Data files.
        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.npy')

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            # Pre process data.
            print('Reading text file...')
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            # Load pre-processed files.
            print('Loading pre-processed files...')
            self.load_preprocessed(vocab_file=vocab_file, tensor_file=tensor_file)

        # Create batches & set batch pointer to 0.
        self.create_batches()
        self.pointer = 0

    def preprocess(self, input_file: str, vocab_file: str, tensor_file: str):
        """Pre-process dataset. Converts text data into numeric format.

        Arguments:
            input_file {str} -- Input file containing the text data.
            vocab_file {str} -- File where all unique characters/vocab in the dataset is stored.
            tensor_file {str} -- File where the numeric representation of dataset is saved.
        """

        # Read input.txt contents.
        with codecs.open(input_file, mode='r', encoding=self.encoding) as f:
            data = f.read()

        # Dictionary of each character & character count.
        counter = collections.Counter(data)

        # List of tuples: [(' ', 14), ('e', 18), ..., ('m', 1)]
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        # Extract the characters: returns tuple of chars.
        self.chars, _ = zip(*count_pairs)

        # Number of characters in the dataset.
        self.vocab_size = len(self.chars)

        # Mapping from char to id or vocab.
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        with open(vocab_file, mode='wb') as f:
            pickle.dump(self.vocab, f)

        # Numeric representation of dataset.
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file: str, tensor_file: str):
        """Load pre-processed data and set all necessary values.

        Arguments:
            vocab_file {str} -- Files where all unique characters/vocab in the dataset is stored.
            tensor_file {str} -- File where the numeric representation of dataset is saved.
        """
        # Open vocab file & read in all unique chars/vocab.
        with open(vocab_file, mode='rb') as f:
            self.chars = pickle.load(f)

        # Create vocab dictionary. & size of all unique characters.
        self.vocab_size = len(self.chars)
        self.vocab = {c: i for i, c in enumerate(self.chars)}

        # Numeric representation of dataset.
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        """Create training  batch.

        Raises:
            AssertionError -- Not enough data. Make batch_size & seq_len smaller.
        """
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # When self.tensor (data) is too small.
        if self.num_batches == 0:
            raise AssertionError("Not enough data. Make batch_size & seq_length small.")

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]

        x_data = self.tensor
        y_data = np.copy(self.tensor)

        y_data[:-1] = x_data[1:]
        y_data[-1] = x_data[0]

        self.x_batch = np.split(np.reshape(x_data, (self.batch_size, -1)),
                                self.num_batches, axis=1)
        self.y_batch = np.split(np.reshape(y_data, (self.batch_size, -1)),
                                self.num_batches, axis=1)

    def next_batch(self):
        """Generate next training batch.

        Returns:
            {list} -- Next batch with shape [batch_size, -1]
        """
        x, y = self.x_batch[self.pointer], self.y_batch[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        """Resets batch pointer to 0."""
        self.pointer = 0
