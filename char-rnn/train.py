import argparse


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
    pass


if __name__ == '__main__':
    main()
