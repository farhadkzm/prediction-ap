import collections
from datetime import datetime

import numpy as np
import tensorflow as tf

import data_reader

# ACCEPT_TIME,ADDRESS_CLUSTER,ARTICLE_ID,
# CONTRACT_ID,DELIVERY_DATE,DELIVERY_TIME,DELIVERY_WEEKDAY,
# DEVICE_USER_ID,EVENT_TIMESTAMP,NUMERIC_TIME,PRODUCT_CD,
# RECEIVER_DPID,RECEIVER_SUBURB,SCAN_EVENT_CD,SCAN_SOURCE_DEVICE,
# SIDE,THOROUGHFARE_TYPE_CODE,USER_ROLE,WORK_CENTRE_CD

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
    'RECEIVER_DPID',
    'THOROUGHFARE_TYPE_CODE',
    'SIDE',
     'PRODUCT_CD',
]

bucketised_columns = [
    'DELIVERY_WEEKDAY',
    'RECEIVER_SUBURB',
    'RECEIVER_DPID',
    'THOROUGHFARE_TYPE_CODE',

    'CONTRACT_ID',
    'USER_ROLE',
    'DEVICE_USER_ID',
    'SCAN_EVENT_CD',
     'PRODUCT_CD',

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
    is_dnn = True
    # building columns
    delivery_weekday = tf.contrib.layers.sparse_column_with_integerized_feature('DELIVERY_WEEKDAY', bucket_size=1000)
    suburb = tf.contrib.layers.sparse_column_with_hash_bucket('RECEIVER_SUBURB', hash_bucket_size=1000)
    delivery_weekday__x__suburb = tf.contrib.layers.crossed_column([delivery_weekday, suburb],
                                                                   hash_bucket_size=int(1e6))

    receiver_dpid = tf.contrib.layers.sparse_column_with_hash_bucket('RECEIVER_DPID', hash_bucket_size=1e4)
    thoroughfare_type_code = tf.contrib.layers.sparse_column_with_hash_bucket('THOROUGHFARE_TYPE_CODE',
                                                                              hash_bucket_size=1e4)
    rec_dpid__x__thoroughfare = tf.contrib.layers.crossed_column([receiver_dpid, thoroughfare_type_code],
                                                                  hash_bucket_size=int(1e6))
    bucketised_features = [
        delivery_weekday,
        suburb,
        delivery_weekday__x__suburb,

        receiver_dpid,
        thoroughfare_type_code,
        rec_dpid__x__thoroughfare,

        tf.contrib.layers.sparse_column_with_hash_bucket('CONTRACT_ID', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('USER_ROLE', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('DEVICE_USER_ID',
                                                         hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_hash_bucket('SCAN_EVENT_CD', hash_bucket_size=1000),

        tf.contrib.layers.sparse_column_with_integerized_feature('PRODUCT_CD', bucket_size=1000),

    ]
    real_value_features = [
        tf.contrib.layers.real_valued_column('NUMERIC_ACCEPT_TIME'),
        # convert_real(tf.contrib.layers.real_valued_column('NUMERIC_ACCEPT_TIME'), [i for i in range(24)], is_dnn),
    ]
    return [convert_bucket(feature, is_dnn) for feature in bucketised_features] + real_value_features


def get_data():
    batch_data_size = 300
    num_groups, group_pick_size = 1, 15000
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
                         , lambda predicted: custom_error(predicted, cv_set, label_column))

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


def custom_error(predicted, cv_set, label_column):
    if len(predicted) != len(cv_set.index):
        raise Exception("size of predicted result is not as cv set")

    prediction_diff = predicted - np.array(cv_set[label_column].values)
    analysed_errors = (abs(prediction_diff + .75) > 2.0)
    error_indices = np.where(analysed_errors > 0)
    errors = cv_set.iloc[error_indices].copy()
    errors['PREDICTION_DIFF'] = prediction_diff[error_indices]

    correct_indices = np.where(analysed_errors == 0)
    corrects = cv_set.iloc[correct_indices].copy()
    corrects['PREDICTION_DIFF'] = prediction_diff[correct_indices]

    file_path = "./logdir/errs/{}".format(datetime.now().strftime('%Y_%m_%d__%H_%M_%S_%f'))
    errors.to_csv(file_path + "-errors.csv")
    corrects.to_csv(file_path + "-corrects.csv")
    return analysed_errors.sum()
