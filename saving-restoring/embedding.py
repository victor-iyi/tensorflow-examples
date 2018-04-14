import os
import tensorflow as tf

# Directories & files
save_dir = os.path.join('saved/tf-save', 'embedding')
tensorboard_dir = os.path.join(save_dir, 'tensorboard')
logdir = os.path.join(tensorboard_dir, 'log')
model_dir = os.path.join(save_dir, 'models')
model_path = os.path.join(model_dir, 'model.ckpt')


def batch_text(corpus, batch_size, seq_length):
    """Batch generator function for generating the next
    batch of a given corpus during training.

    Args:
        corpus (str):
            Data corpus.
        batch_size (int):
            Size of the generated batch.
        seq_length (int):
            Length of a single sequence in dataset.

    Raises:
        ValueError:
            seq_length >= len(corpus)

    Yeilds:
        x, y
        Next training batch with it's corresponding label.
    """
    if seq_length >= len(corpus):
        raise ValueError('seq_length >= len(corpus): {}>={}'.format(
            seq_length, len(corpus)))

    sequences = [corpus[i: i + seq_length]
                 for i in range(len(corpus) - seq_length)]
    ys = [corpus[i: i + 1] for i in range(seq_length, len(corpus))]

    for i in range(0, len(sequences), batch_size):
        x = sequences[i: i + batch_size]
        y = ys[i: i + batch_size]

        yield x, y
