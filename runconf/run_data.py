import collections
import delivery_data_reader
import tensorflow as tf


def get_data(feature_names, bucketised_columns,
             num_groups, group_pick_size):
    RunData = collections.namedtuple('RunData', ['train_x',
                                                 'test_y',
                                                 'cv_y',
                                                 'train_batch_input_fn',
                                                 'train_eval_input_fn',
                                                 'cv_eval_input_fn',
                                                 'test_eval_input_fn',
                                                 ])
    dics = delivery_data_reader.initialise_data('./data/train.csv', '', feature_names, bucketised_columns,
                                                num_groups, group_pick_size)

    run_config = RunData(dics['train_x']
                         , dics['test_y']
                         , dics['cv_y']
                         , lambda index, batch_size: train_input_fn(dics['train_x'], dics['train_y'], index, batch_size)
                         , lambda: input_fn(dics['train_x'], dics['train_y'])
                         , lambda: input_fn(dics['cv_x'], dics['cv_y'])
                         , lambda: input_fn(dics['test_x'], dics['test_y']))

    return run_config


def input_fn(input_x, input_y):
    continuous_cols = {
        feature_name: tf.constant(input_x[feature_name].values, shape=[len(input_x[feature_name].values)],
                                  verify_shape=True) for feature_name in input_x.columns}
    label = tf.constant(input_y.values, shape=[len(input_y.values)], verify_shape=True)

    return continuous_cols, label


def train_input_fn(train_x, train_y, index, batch_size):
    start = index * batch_size
    end = start + batch_size
    return input_fn(train_x[start:end], train_y[start:end])
