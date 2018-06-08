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


def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest="frozen_file", type=str, default='saved/frozen.pb',
                        help='Frozen model file to import')
    parser.add_argument('-m', dest='model_dir', type=str, default='saved/model',
                        help='Model directory containing checkpoints.')

    args = parser.parse_args()

    print('{0}\n{1:^45}\n{0}'.format('-'*45, 'Parameters'))
    for k, v in vars(args).items():
        print('{0:<15} = {0:>20}'.format(k, v))
    print('{0}'.format('-'*45))

    main(args)
