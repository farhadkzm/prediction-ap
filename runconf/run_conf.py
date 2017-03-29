import collections
import logging

import tensorflow as tf

import run_data


def get_run_conf():
    # all columns
    # ACCEPT_TIME,ADDRESS_CLUSTER,ARTICLE_ID,
    # CONTRACT_ID,DELIVERY_DATE,DELIVERY_TIME,DELIVERY_WEEKDAY,
    # DEVICE_USER_ID,EVENT_TIMESTAMP,NUMERIC_TIME,PRODUCT_CD,
    # RECEIVER_DPID,RECEIVER_SUBURB,SCAN_EVENT_CD,SCAN_SOURCE_DEVICE,
    # SIDE,THOROUGHFARE_TYPE_CODE,USER_ROLE,WORK_CENTRE_CD
    feature_names = [
        'ACCEPT_TIME',
        'NUMERIC_TIME',
        'DELIVERY_WEEKDAY',
        'CONTRACT_ID',
        'USER_ROLE',
        'DEVICE_USER_ID',
        'SCAN_EVENT_CD',
        'PRODUCT_CD',
        'RECEIVER_SUBURB',
        'THOROUGHFARE_TYPE_CODE',
        'SIDE'
    ]

    bucketised_columns = [
        'DELIVERY_WEEKDAY',
        'CONTRACT_ID',
        'USER_ROLE',
        'DEVICE_USER_ID',
        'SCAN_EVENT_CD',
        'PRODUCT_CD',
        'RECEIVER_SUBURB',
        'THOROUGHFARE_TYPE_CODE',
        'SIDE'
    ]

    run_name = 'NN_run_conf'
    num_groups, group_pick_size = 5, 6000

    evaluate_steps = 100
    batch_data_size = 300

    data_config = run_data.get_data(feature_names, bucketised_columns,
                                    num_groups, group_pick_size)
    train_x = data_config.train_x

    layers = [len(train_x.columns), 70, 50, 30]

    logging.debug('Creating NN with %s', layers)
    # estimator = tf.contrib.learn.DNNRegressor(
    #     feature_columns=[tf.contrib.layers.real_valued_column(col_name) for col_name in
    #                      train_x.columns],
    #     hidden_units=layers)
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column(col_name) for col_name in
                         train_x.columns],
        hidden_units=layers,
        activation_fn=tf.nn.tanh,
        optimizer=tf.train.AdagradOptimizer(
            learning_rate=0.0005,
            # l1_regularization_strength=0.001
        ))

    RunConf = collections.namedtuple('RunConf', ['feature_names',
                                                 'bucketised_columns',
                                                 'num_groups',
                                                 'group_pick_size',
                                                 'evaluate_steps',
                                                 'batch_data_size',
                                                 'estimator',
                                                 'run_name',
                                                 'data_config',

                                                 ])

    return RunConf(
        feature_names
        , bucketised_columns
        , num_groups
        , group_pick_size
        , evaluate_steps
        , batch_data_size
        , estimator
        , run_name
        , data_config)
