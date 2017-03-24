import logging

import pandas as pd

import data_reader


def __convert_time_to_float(value):
    digits = value.split(':')
    return float(digits[0]) + float(digits[1]) * 1 / 60


converters_map = {'ACCEPT_NUMERIC': ('ACCEPT_TIME', __convert_time_to_float)}


def __read_data(data_path, feature_names, bucketised_columns, num_groups, group_pick_size):
    return data_reader.read_prepared_data(data_path, feature_names, num_groups, group_pick_size,
                                          bucketised_columns, converters_map)


def initialise_data(train_path, out_dir_path,
                    feature_names, bucketised_columns,
                    num_groups, group_pick_size,
                    write=False):
    # reading data
    train_set, cv_set, test_set = __read_data(train_path,
                                              feature_names, bucketised_columns,
                                              num_groups, group_pick_size)

    train_x, train_y = data_reader.split_x_y(train_set, 'NUMERIC_TIME')
    cv_x, cv_y = data_reader.split_x_y(cv_set, 'NUMERIC_TIME')
    test_x, test_y = data_reader.split_x_y(test_set, 'NUMERIC_TIME')

    data_reader.make_compatible(cv_x, train_x)
    data_reader.make_compatible(test_x, train_x)

    # writing data to file
    files = {'test_x': test_x,
             'test_y': test_y,
             'cv_x': cv_x,
             'cv_y': cv_y,
             'train_x': train_x,
             'train_y': train_y,
             }
    if write:
        for fl in files.keys():
            df = files[fl]
            df.to_csv("{}/{}.csv".format(out_dir_path, fl))

    return files


def read_files(out_dir_path):
    x = ['test_x', 'cv_x', 'train_x', ]
    y = ['test_y', 'cv_y', 'train_y', ]
    fls = {}
    logging.debug('Reading directory %s', out_dir_path)
    for fl in x:
        logging.debug('Reading file %s...', fl)
        df = pd.read_csv("{}/{}.csv".format(out_dir_path, fl))
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        fls[fl] = df
    for fl in y:
        logging.debug('Reading file %s...', fl)
        ser = pd.read_csv("{}/{}.csv".format(out_dir_path, fl), usecols=[1], squeeze=True, header=False)
        fls[fl] = ser

    return fls
