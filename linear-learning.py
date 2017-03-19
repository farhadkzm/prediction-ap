import tensorflow as tf
import numpy as np
import delivery_data_reader
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

max_row_size = 50000

learning_rate = 0.002

fit_max_steps = 2000
evaluate_steps = 100
batch_data_size = 500

# semi-constant variables
log_path = './logdir/tf_logs/' + datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# delivery_data_reader.initialise_data('./data/train.csv','./data/test.csv','./data/gen')

dics = delivery_data_reader.read_files('./data/gen')

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
    logging.debug('Reading %s rows for step %s', batch_size, index + 1)
    start = index * batch_size
    end = start + batch_size
    return input_fn(train_x[start:end], train_y[start:end])


def test_input_fn():
    return input_fn(test_x, test_y)


# estimator = tf.contrib.learn.LinearRegressor(feature_columns=
#                                              [tf.contrib.layers.real_valued_column(col_name) for col_name in
#                                               train_x.columns])
#
estimator = tf.contrib.learn.DNNRegressor(
    feature_columns=[tf.contrib.layers.real_valued_column(col_name) for col_name in
                     train_x.columns],
    hidden_units=[len(train_x.columns), 512, 256])


def time_window_error(summary):
    predicted = estimator.predict(input_fn=test_input_fn)

    predicted = np.array(list(predicted))
    error_count = (abs(predicted - np.array(test_y.values)) > 2).sum()
    error = float(error_count) / len(predicted)
    logging.debug('Window Error for TEST: %s/%s ratio: %s', error_count, len(predicted), error)
    summary.value.add(tag='outliers', simple_value=error)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_path, sess.graph)

    for i in range(total_iterations):
        logging.debug('Starting fitting for step %s', i + 1)
        estimator.partial_fit(input_fn=(lambda: train_input_fn(i, batch_data_size)))

        if True:
            logging.debug('Aggregating statistics...')
            summary = tf.Summary()

            # getting losses for test and train
            logging.debug('Loss Eval for TRAIN')
            train_evaluation = estimator.evaluate(input_fn=train_input_fn, steps=evaluate_steps)
            summary.value.add(tag='train_loss', simple_value=train_evaluation['loss'])

            logging.debug('Loss Eval for TEST')
            test_evaluation = estimator.evaluate(input_fn=test_input_fn, steps=evaluate_steps)
            summary.value.add(tag='test_loss', simple_value=test_evaluation['loss'])

            # getting window_error for test
            logging.debug('Window Eval for Test')
            time_window_error(summary)
            writer.add_summary(summary, global_step=i)
            logging.debug('writing metrics')
            writer.flush()

    writer.close()
print 'Processed finished'
