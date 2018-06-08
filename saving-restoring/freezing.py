"""Freezing a model in TensorFlow simply means to pack our parameters into one
single file instead of 3 different files we have during checkpoints for each iteration.

    Steps For Freezing a model:

    - Retrieve our saved graph: we need to load the previously saved meta graph in
    the default graph and retrieve it's graph_def (the ProtoBuff definition of our graph).

    - Restore the weights: we start a tf.Session & restore the weights of our graph inside
    that Session.

    - Remove all metadata useless for inference: Here, TensorFlow helps us with a nice helper
    function which graph just what is needed in your graph to perform inference and returns
    what we will call our new "frozen graph_def".

    - Save it to the disk, finally we will serialize our frozen graph_def ProtoBuff and dump
    it to disk.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: freezing.py
     Created on 30 May, 2018 @ 8:20 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import argparse
import os

import tensorflow as tf

# The original `freeze_graph` function
# from tensorflow.python.tools.freeze_graph import freeze_graph

# Current parent directory.
cwd = os.path.dirname(__file__)


def freeze_graph(ckpt_dir: str, output_nodes: list):
    """Extract the sub graph defined by the `output_nodes` and convert
    all it's variables to constant.

    Arguments:
        ckpt_dir {str} -- Root folder containing checkpoint files.
        output_nodes {list} -- A list containing all the output node's names.

    Returns:
        {tf.GraphDef} -- Output graph definition.
    """
    # Make sure export directory exists.
    if not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(("Export directory doesn't exists or isn't a directory."
                                  "Please specify an export directory. {}").format(ckpt_dir))

    # Make sure there's at list one output node name.
    assert output_nodes, 'You need to supply a list of output node names.'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
    input_ckpt = ckpt.model_checkpoint_path


def main(args):
    print(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', dest="ckpt_dir", type=str, default='../saved/rnn',
                        help='Path to a directory containing a TensorFlow checkpoint model.')
    parser.add_argument('-f', dest="frozen_file", type=str, default='frozen.pb',
                        help='Frozen model filename to import (must have a .pb extension).')
    parser.add_argument('-o', dest='output_nodes', type=str, default='',
                        help='Output nodes (separated by comma).')

    args = parser.parse_args()

    # Print out the parsed command line arguments.
    print('{0}\n{1:^45}\n{0}'.format('-'*45, 'Parameters'))
    for k, v in vars(args).items():
        print('{0:<15} = {0:>20}'.format(k, v))
    print('{0}'.format('-'*45))

    main(args)
