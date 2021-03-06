import collections
import logging

import tensorflow as tf


def get_run_conf(feature_columns):
    run_name = 'linear_boundries_on_real_values'

    evaluate_steps = 10

    layers = [1000, 512, 50]

    logging.debug('Creating NN with %s', layers)
    # estimator = tf.contrib.learn.LinearRegressor(
    #     feature_columns=feature_columns)
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=layers,
        # activation_fn=tf.nn.tanh,
        # optimizer=tf.train.ProximalAdagradOptimizer(
        #     learning_rate=0.1,
        #     l1_regularization_strength=0.0,
        #     l2_regularization_strength=0.0
        # )
    )

    RunConf = collections.namedtuple('RunConf', [

        'evaluate_steps',
        'estimator',
        'run_name',

    ])

    return RunConf(

        evaluate_steps
        , estimator
        , run_name
    )
