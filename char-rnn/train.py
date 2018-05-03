import argparse
import os
import pickle
import time

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

    # Checkpoint state.
    ckpt = None

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
    model = Model(args, training=True)

    # Start TensorFlow session. (with the default graph).
    with tf.Session() as sess:
        # Summary for Tensorboard.
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.logdir, time.strftime("%Y-%m-%d-%H-%M-%S-%p")),
                                       graph=sess.graph)
        writer.add_graph(graph=sess.graph)

        # Initialize global variables.
        sess.run(tf.global_variables_initializer())

        # Saver object for all global variables.
        saver = tf.train.Saver(var_list=tf.global_variables())

        # Restore model from checkpoint.
        if args.init_from is not None:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

        # TRAINING LOOP.
        for epoch in range(args.num_epochs):
            # NOTE: Surrounded with try-except in case training was force-stopped.
            try:
                # Update Model's learning rate.
                sess.run(tf.assign(model.lr, value=args.learning_rate * (args.decay_rate ** epoch)))

                # Reset mini batch pointer.
                data_loader.reset_batch_pointer()

                # Initial state.
                state = sess.run(model.initial_state)

                for batch in range(data_loader.num_batches):
                    # Record start time for current batch.
                    start = time.time()

                    # Get the next mini batch.
                    X, y = data_loader.next_batch()

                    feed_dict = {model.input_data: X, model.targets: y}

                    for i, (c, h) in enumerate(model.initial_state):
                        feed_dict[c] = state[i].c
                        feed_dict[h] = state[i].h

                    # Train the model.
                    _, _loss, _global, _summary, state = sess.run([model.train_op, model.loss, model.global_step,
                                                                   summaries, model.final_state], feed_dict=feed_dict)

                    writer.add_summary(summary=_summary, global_step=_global)

                    end = time.time()
                    batch_count = epoch * data_loader.num_batches + batch

                    # Log progress.
                    print("\r{:,} of {:,} | global: {:,} Loss: {} time/batch: {}"
                          .format(batch_count, args.num_epochs, _global, _loss, end - start), end="")

                    # Save model at intervals.
                    if batch_count % args.save_every == 0 or (
                                    epoch == args.num_epochs - 1 and batch == data_loader.num_batches - 1):
                        save_path = os.path.join(args.save_dir, "model.ckpt")
                        saver.save(sess=sess, save_path=save_path, global_step=model.global_step)

                        print("\nModel saved to {}\n".format(save_path))

                """# !- end batch"""
            except KeyboardInterrupt:
                print('\nTraining interrupted by user. Saving...')

                save_path = os.path.join(args.save_dir, "model.ckpt")
                saver.save(sess=sess, save_path=save_path,
                           global_step=model.global_step)

                print("Model saved to {}\n".format(save_path))

                # End training.
                break

        # !- end epoch
        print("\n\nOverall training count = {}".format(sess.run(model.global_step)))


if __name__ == '__main__':
    main()
