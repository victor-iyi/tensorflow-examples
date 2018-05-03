"""
  @author 
    Victor I. Afolabi
    Artificial Intelligence & Software Engineer.
    Email: javafolabi@gmail.com
    GitHub: https://github.com/victor-iyiola
  
  @project
    File: pre-made_estimator.py
    Created on 03 May, 2018 @ 7:13 PM.
    
  @license
    MIT License
    Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import argparse

import tensorflow as tf
from .iris_data import train_input_fn, eval_input_fn, load_data


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini batch size.")
    parser.add_argument("--train_steps", type=int, default=10,
                        help="Number of training iterations.")

    args = parser.parse_args(args=argv[1:])

    (train_X, train_y), (test_X, test_y) = load_data()

    feature_cols = []
    for key in train_X.keys():
        feature_cols.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(hidden_units=[10, 10],
                                            feature_columns=feature_cols,
                                            n_classes=3)

    # Train the classifier.
    classifier.train(input_fn=lambda: train_input_fn(train_X, train_y, args.batch_size),
                     steps=args.train_step)

    # Evaluate model's accuracy.
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_X, test_y, args.batch_size))

    print("Test set accuracy = {accuracy:.3f}\n".format(**eval_result))

    label_names = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    # Make predictions on new dataset.
    predictions = classifier.predict(input_fn=lambda: eval_input_fn(predict_x, batch_size=args.batch_size))

    for pred, l_name in zip(predictions, label_names):
        class_id = pred['class_id'][0]  # Class id 0, 1 or 2.
        prob = pred['probabilities']['class_id']  # Confidence probability.

        print('Prediction is "{}"({:.2%}), expected: "{}"'.format(predict_x[class_id], prob, l_name))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
