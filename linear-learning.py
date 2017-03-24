import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import delivery_data_reader

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

num_groups, group_pick_size = 10, 3000

evaluate_steps = 100
batch_data_size = 100

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# semi-constant variables
log_path = './logdir/tf_logs/' + datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

dics = delivery_data_reader.initialise_data('./data/train.csv', './data/gen', feature_names, bucketised_columns,
                                            num_groups, group_pick_size)

# dics = delivery_data_reader.read_files('./data/gen')

test_x = dics['test_x']
test_y = dics['test_y']
cv_x = dics['cv_x']
cv_y = dics['cv_y']
train_x = dics['train_x']
train_y = dics['train_y']

train_rows = len(train_x.index)
total_iterations = int(train_rows / batch_data_size)


def input_fn(input_x, input_y):
    continuous_cols = {
        feature_name: tf.constant(input_x[feature_name].values, shape=[len(input_x[feature_name].values)],
                                  verify_shape=True) for feature_name in input_x.columns}
    label = tf.constant(input_y.values, shape=[len(input_y.values)], verify_shape=True)

    return continuous_cols, label


def train_input_fn(index=0, batch_size=train_rows):
    logging.debug('Reading the next %s rows', batch_size)
    start = index * batch_size
    end = start + batch_size
    return input_fn(train_x[start:end], train_y[start:end])


def train_eval_input_fn():
    return input_fn(train_x, train_y)


def test_input_fn():
    return input_fn(test_x, test_y)


# estimator = tf.contrib.learn.LinearRegressor(
#     feature_columns=[tf.contrib.layers.real_valued_column(col_name) for col_name in
#                      train_x.columns])
#

estimator = tf.contrib.learn.DNNRegressor(
    feature_columns=[tf.contrib.layers.real_valued_column(col_name) for col_name in
                     train_x.columns],
    hidden_units=[len(train_x.columns), 512, 256])


def time_window_error(summary_op):
    predicted = estimator.predict(input_fn=test_input_fn)

    predicted = np.array(list(predicted))
    error_count = (abs(predicted - np.array(test_y.values)) > 2).sum()
    error = float(error_count) / len(predicted)
    logging.debug('Window Error for TEST: %s/%s ratio: %s', error_count, len(predicted), "{0:.2f}".format(error))
    summary_op.value.add(tag='outliers', simple_value=error)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_path, sess.graph)

    for i in range(total_iterations):
        logging.debug('Starting fitting for step %s/%s', i + 1, total_iterations)
        estimator.partial_fit(input_fn=(lambda: train_input_fn(i, batch_data_size)))

        if True:
            logging.debug('Aggregating statistics...')
            summary = tf.Summary()

            # getting losses for test and train
            logging.debug('Processing Train Loss')
            train_evaluation = estimator.evaluate(input_fn=train_eval_input_fn, steps=evaluate_steps)
            summary.value.add(tag='train_loss', simple_value=train_evaluation['loss'])

            logging.debug('Processing Test Loss')
            test_evaluation = estimator.evaluate(input_fn=test_input_fn, steps=evaluate_steps)
            summary.value.add(tag='test_loss', simple_value=test_evaluation['loss'])

            # getting window_error for test
            logging.debug('Processing Test Outliers')
            time_window_error(summary)
            writer.add_summary(summary, global_step=i)
            logging.debug('writing metrics')
            writer.flush()

    writer.close()
print 'Processed finished'
