import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, default='data/pycode',
                        help='Data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='Directory where checkpoints are stored.')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Tensorboard log directory.')


if __name__ == '__main__':
    main()
