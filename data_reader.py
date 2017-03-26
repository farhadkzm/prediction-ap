import logging

import numpy as np
import pandas as pd

import etl


def read_csv(data_path, imported_cols, limit=None):
    df = pd.read_csv(data_path, usecols=imported_cols)
    total_rows = len(df.index)

    df.dropna(inplace=True)
    removed_rows = total_rows - len(df.index)
    logging.debug('After dropping NA. Rows: total %s - Removed %s', total_rows, removed_rows)

    return df


def __add_unique_values_as_columns(df, column):
    for unique_val in df[column].unique():
        new_col_name = column + '_' + filter(str.isalnum, str(unique_val))
        df[new_col_name] = (df[column] == unique_val).astype(np.float32)


def bucketised_columns(df, columns):
    for col in columns:
        __add_unique_values_as_columns(df, col)

    # removing bucketised columns
    df.drop(labels=columns, axis=1, inplace=True)


def convert_columns(df, converters):
    for column_name in converters.keys():
        existing_column, converter = converters[column_name]
        df[column_name] = df.apply(lambda row: converter(row[existing_column]), axis=1)
        df.drop(labels=[existing_column], axis=1, inplace=True)


def __prepare_data(df, bck_columns=None, converters=None):
    total_columns = len(df.columns)
    logging.debug('Bucketising columns')
    if bck_columns is not None:
        bucketised_columns(df, bck_columns)

    logging.debug('Added columns after bucketisation: %s', len(df.columns) - total_columns)

    if converters is not None:
        logging.debug('Applying converters on columns')
        convert_columns(df, converters)


def read_prepared_data(data_path, imported_cols=None, num_groups=10, group_pick_size=3000, bck_columns=None,
                       converters=None):
    logging.debug('Reading csv file %s', data_path)
    df = read_csv(data_path, imported_cols)

    groupby_col = 'RECEIVER_SUBURB'
    working_set = etl.smart_split(df, groupby_col, num_groups, group_pick_size)

    grouped = working_set.groupby(groupby_col)

    testcv_ratio = .4

    testcv_group_pick_size = int(group_pick_size * testcv_ratio)
    train_group_pick_size = group_pick_size - testcv_group_pick_size

    logging.debug('Grouping %s. Items in each group %s. CV/Test items %s. Train items %s', groupby_col, group_pick_size,
                  testcv_group_pick_size, train_group_pick_size)

    testcv_set = grouped.head(testcv_group_pick_size)
    train_set = grouped.tail(train_group_pick_size).copy()

    test_group_pick = int(testcv_group_pick_size * .5)
    cv_group_pick = testcv_group_pick_size - test_group_pick

    testcv_grouped = testcv_set.groupby(groupby_col)
    test_set = testcv_grouped.head(test_group_pick).copy()
    cv_set = testcv_grouped.tail(cv_group_pick).copy()

    logging.debug('Preparing test set...')
    __prepare_data(test_set, bck_columns, converters)
    logging.debug('Preparing cv set...')
    __prepare_data(cv_set, bck_columns, converters)
    logging.debug('Preparing train set...')
    __prepare_data(train_set, bck_columns, converters)

    make_compatible(cv_set, train_set)
    make_compatible(test_set, train_set)
    return train_set, cv_set, test_set


def make_compatible(cv, train):
    cols_to_add_train = set(cv.columns) - set(train)
    cols_to_add_cv = set(train.columns) - set(cv)

    for col in cols_to_add_cv:
        cv[col] = np.float32(0)

    for col in cols_to_add_train:
        train[col] = np.float32(0)


def split_x_y(df, label_column):
    y = df[label_column]
    df.drop(label_column, axis=1, inplace=True)
    x = df
    return x, y
