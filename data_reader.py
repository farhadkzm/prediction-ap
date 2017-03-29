import logging

import pandas as pd


def read_prepared_data(data_path, imported_cols=None, num_groups=10, group_pick_size=3000):
    logging.debug('Reading csv file %s', data_path)
    df = pd.read_csv(data_path, usecols=imported_cols)

    groupby_col = 'RECEIVER_SUBURB'
    working_set = smart_split(df, groupby_col, num_groups, group_pick_size)

    logging.debug('Converting columns...')
    generate_inferred_columns(working_set)
    grouped = working_set.groupby(groupby_col)

    testcv_ratio = .4

    testcv_group_pick_size = int(group_pick_size * testcv_ratio)
    train_group_pick_size = group_pick_size - testcv_group_pick_size

    logging.debug('Grouping column %s. Group size %s. CV/Test items %s. Train items %s', groupby_col, group_pick_size,
                  testcv_group_pick_size, train_group_pick_size)

    testcv_set = grouped.head(testcv_group_pick_size)
    train_set = grouped.tail(train_group_pick_size).copy()

    test_group_pick = int(testcv_group_pick_size * .5)
    cv_group_pick = testcv_group_pick_size - test_group_pick

    testcv_grouped = testcv_set.groupby(groupby_col)
    test_set = testcv_grouped.head(test_group_pick).copy()
    cv_set = testcv_grouped.tail(cv_group_pick).copy()
    logging.debug('Rows for sets, Train %s, CV %s, Test %s', len(train_set.index), len(cv_set.index),
                  len(test_set.index))
    return train_set, cv_set, test_set


def smart_split(df, groupby_col, number_of_group=10, group_pick_size=1000):
    groups = df.groupby(groupby_col).size()
    g1k = groups[groups > group_pick_size]
    g1k = list(g1k.index)
    df1k = df[df[groupby_col].isin(g1k)]

    total_rows = number_of_group * group_pick_size
    working_set = df1k.groupby(groupby_col).head(group_pick_size).sort_values(groupby_col).iloc[
                  0:total_rows]

    return working_set


def generate_inferred_columns(df):
    df['NUMERIC_ACCEPT_TIME'] = df.apply(lambda row: __convert_time_to_float(row['ACCEPT_TIME']), axis=1)

    # The following usually used for categorical predictions
    # Theory 1: map NUMERIC_TIME to a 2h window starting from last even hour
    # Theory 2: map NUMERIC_TIME to a 2h window starting from the current hour
    # Theory 3: map NUMERIC_TIME to a nearest hour

    # Theory 4: instead of NUMERIC_TIME for y, use DELIVERY_TIME - NUMERIC_ACCEPT_TIME as y
    df['DIFF_NUMERIC_TIME'] = df.apply(lambda row: row['NUMERIC_TIME'] - row['NUMERIC_ACCEPT_TIME'], axis=1)

    pass


def __convert_time_to_float(value):
    digits = value.split(':')
    return float(digits[0]) + (float(digits[1]) / 60.0)
