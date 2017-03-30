import collections

import numpy as np
import tensorflow as tf

import data_reader

feature_names = [
    'ACCEPT_TIME',  # this will be converted to NUMERIC_ACCEPT_TIME
    'NUMERIC_TIME',
    'DELIVERY_WEEKDAY',
    'CONTRACT_ID',
    'USER_ROLE',
    'DEVICE_USER_ID',
    'SCAN_EVENT_CD',
    'PRODUCT_CD',
    'RECEIVER_SUBURB',
    'THOROUGHFARE_TYPE_CODE',
    'SIDE',
    'PRODUCT_CD',
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
    'SIDE',
]


def convert_bucket(feature, is_dnn=False):
    if is_dnn:
        return tf.contrib.layers.embedding_column(feature, dimension=8)
    else:
        return feature


def convert_real(feature, boundaries, is_dnn=False):
    if is_dnn:
        return feature
    else:
        return tf.contrib.layers.bucketized_column(feature, boundaries=boundaries)


def create_features():
    is_dnn = False
    # building columns
    delivery_weekday = tf.contrib.layers.sparse_column_with_integerized_feature('DELIVERY_WEEKDAY', bucket_size=1000)
    suburb = tf.contrib.layers.sparse_column_with_hash_bucket('RECEIVER_SUBURB', hash_bucket_size=1000)
    delivery_weekday__x__suburb = tf.contrib.layers.crossed_column([delivery_weekday, suburb],
                                                                   hash_bucket_size=int(1e6))

    bucketised_features = [
        delivery_weekday,
        suburb,
        # delivery_weekday__x__suburb,

        tf.contrib.layers.sparse_column_with_hash_bucket('CONTRACT_ID', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('USER_ROLE', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('DEVICE_USER_ID',
                                                         hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('SCAN_EVENT_CD', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_integerized_feature('PRODUCT_CD', bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('THOROUGHFARE_TYPE_CODE',
                                                         hash_bucket_size=1000),
    ]
    real_value_features = [
        tf.contrib.layers.real_valued_column('NUMERIC_ACCEPT_TIME'),
        # convert_real(tf.contrib.layers.real_valued_column('NUMERIC_ACCEPT_TIME'), [i for i in range(24)], is_dnn),
    ]
    return [convert_bucket(feature, is_dnn) for feature in bucketised_features] + real_value_features


def get_data():
    batch_data_size = 300
    num_groups, group_pick_size = 5, 10000
    train_set, cv_set, test_set = data_reader.read_prepared_data('./data/train.csv', feature_names, num_groups,
                                                                 group_pick_size)

    total_iterations = int(len(train_set.index) / batch_data_size)

    RunData = collections.namedtuple('RunData', [
        'feature_columns',
        'total_iterations',
        'train_batch_input_fn',
        'train_eval_input_fn',
        'cv_eval_input_fn',
        'custom_error',
    ])

    label_column = 'NUMERIC_TIME'
    # label_column = 'DIFF_NUMERIC_TIME'
    run_config = RunData(create_features()
                         , total_iterations
                         , lambda index: train_input_fn(train_set, train_set[label_column], index, batch_data_size)
                         , lambda: input_fn(train_set, train_set[label_column])
                         , lambda: input_fn(cv_set, cv_set[label_column])
                         , lambda predicted: custom_error(predicted, cv_set[label_column]))

    return run_config


def input_fn(input_x, input_y):
    continuous_cols = {'NUMERIC_ACCEPT_TIME': tf.constant(input_x['NUMERIC_ACCEPT_TIME'].values,
                                                          shape=[len(input_x['NUMERIC_ACCEPT_TIME'].values)],
                                                          verify_shape=True)}
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(input_x[k].size)],
            values=input_x[k].values,
            dense_shape=[input_x[k].size, 1])
        for k in bucketised_columns}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    label = tf.constant(input_y.values, shape=[len(input_y.values)], verify_shape=True)

    return feature_cols, label


def train_input_fn(train_x, train_y, index, batch_size):
    start = index * batch_size
    end = start + batch_size
    return input_fn(train_x[start:end], train_y[start:end])


def custom_error(predicted, cv_y):
    return (abs(predicted - np.array(cv_y.values)) > 2).sum()
