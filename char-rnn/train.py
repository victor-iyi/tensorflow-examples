import argparse
import os
import pickle

from .suppress import *
from .utils import TextLoader
from .model import Model


def main():
    # Argument parser.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Command line arguments.
    parser.add_argument('--data_dir', type=str, default='data/pycode',
                        help='Data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='Directory where checkpoints are stored.')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Tensorboard log directory.')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='Size of RNN hidden cell state.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers in the network.')
    parser.add_argument('--model', type=str, default='lstm',
                        help='Recurrent architecture: RNN, LSTM, GRU or NAS')
    parser.add_argument('==batch_size', type=int,
                        default=50, help='Mini batch size.')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Recurrent sequence length.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save frequency.')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='Clip gradient at this value.')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='Decay rate for RMSProp.')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='Probability of keeping weights in the hidden layers.')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='Probability of keeping weights in the output layer.')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""Continue training from saved model at this path. 
                        Path must contain files saved by previous training process:
                            'config.pkl'        : Configuration;
                            'chars_vocab.pkl'   : Vocabulary definitions;
                            'checkpoint'        : Paths to model file(s) (created by tf).
                                                  Note: This file contains absolute paths, be careful when 
                                                        moving files around;
                            'model.ckpt-*'      : File(s) with model definition (created by tf)""")

    # Parse the arguments.
    args = parser.parse_args()

    # Call the train function to begin training.
    train(args)


def train(args):
    # Load dataset.
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    data_loader.vocab_size = args.vocab_size

    # Check if training can be continued from previously saved model.
    if args.init_from is not None:
        # Assert all necessary files exists.
        assert os.path.isdir(args.init_from), "{} doesn't exist.".format(args.init_from)

        assert os.path.exists(os.path.join(args.init_from, "config.pkl")), \
            "config.pkl doesn't exist in path {}".format(args.init_from)

        assert os.path.exists(os.path.join(args.init_from, "chars_vocab.pkl")), \
            "chars_vocab.pkl doesn't exist in path {}".format(args.init_from)

        # Get the state of checkpoint to be loaded.
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found!"
        assert ckpt.model_checkpoint_path, "model.ckpt-* not found in path {}".format(args.init_from)

        # Open config file and verify model compatibility.
        with open(os.path.join(args.init_from, "config.pkl"), mode="rb") as f:
            saved_model_args = pickle.load(f)

        # List of meta data that needs to be the same
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]

        for check_me in need_be_same:
            assert vars(saved_model_args)[check_me] == vars(args)[check_me], \
                "Saved model & command line arguments of {} aren't compatible!".format(check_me)

        # Load saved chars & vocab and check for compatibility.
        with open(os.path.join(args.init_from, "chars_vocab.pkl"), mode="rb") as f:
            saved_chars, saved_vocab = pickle.load(f)

        assert saved_chars == data_loader.chars, "Data and character set aren't compatible!"
        assert saved_vocab == data_loader.vocab, "Data and loaded dictionary mappings aren't compatible!"

    # Create save directory if it doesn't exist.
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Write the command line arguments into config file.
    with open(os.path.join(args.save_dir, "config.pkl"), mode="wb") as f:
        pickle.dump(args, f)

    # Save character set and dictionary mappings
    with open(os.path.join(args.save_dir, "chars_vocab.pkl"), mode="wb") as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)

    # Define the model.
    # model = Model(args, training=True)

    # Start TensorFlow session. (with the default graph).


if __name__ == '__main__':
    main()
