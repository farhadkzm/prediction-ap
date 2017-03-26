import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import runconf.run_conf1 as run_conf

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


rc = run_conf.get_run_conf()
feature_names = rc.feature_names

bucketised_columns = rc.bucketised_columns

num_groups, group_pick_size = rc.num_groups, rc.group_pick_size

evaluate_steps = rc.evaluate_steps

batch_data_size = rc.batch_data_size

run_name = rc.run_name
data_config = rc.data_config


# semi-constant variables
log_path = './logdir/tf_logs/' + datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '_' + run_name

train_x = data_config.train_x
test_y = data_config.test_y
train_rows = len(train_x.index)
total_iterations = int(train_rows / batch_data_size)
estimator = rc.estimator


def time_window_error(summary_op):
    predicted = estimator.predict(input_fn=data_config.test_eval_input_fn)

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
        estimator.partial_fit(input_fn=(lambda: data_config.train_batch_input_fn(i, batch_data_size)))

        if True:
            logging.debug('Aggregating statistics...')
            summary = tf.Summary()

            # getting losses for test and train
            logging.debug('Processing Train Loss')
            train_evaluation = estimator.evaluate(input_fn=data_config.train_eval_input_fn, steps=evaluate_steps)
            summary.value.add(tag='train_loss', simple_value=train_evaluation['loss'])

            logging.debug('Processing CV Loss')
            cv_evaluation = estimator.evaluate(input_fn=data_config.cv_eval_input_fn, steps=evaluate_steps)
            summary.value.add(tag='cv_loss', simple_value=cv_evaluation['loss'])

            # getting window_error for test
            logging.debug('Processing Test Outliers')
            time_window_error(summary)
            writer.add_summary(summary, global_step=(i + 1))
            logging.debug('writing metrics')
            writer.flush()

    writer.close()
print 'Processed finished'
