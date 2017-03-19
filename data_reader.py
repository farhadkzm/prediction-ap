import pandas as pd
import numpy as np
import logging


def read_csv(data_path, imported_cols, limit=None):
    df = pd.read_csv(data_path, usecols=imported_cols, nrows=limit)
    total_rows = len(df.index)

    df.dropna(inplace=True)
    removed_rows = total_rows - len(df.index)
    logging.debug('total rows: %s - Removed: %s - file %s', total_rows, removed_rows, data_path)

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


def split_data(df, ratios):
    fragmented = []
    start = 0
    rows = len(df.index)
    for ratio in ratios:
        end = start + int(ratio * rows)
        new_fragment = df.iloc[start: end]
        fragmented.append(new_fragment)
        start += len(new_fragment.index)

    new_fragment = df.iloc[start: None]
    fragmented.append(new_fragment)
    return fragmented


def __read_bucketised_data(data_path, imported_cols, limit=None, bck_columns=None, converters=None):
    df = read_csv(data_path, imported_cols, limit)
    total_columns = len(df.columns)
    logging.debug('Bucketising columns')
    if bck_columns is not None:
        bucketised_columns(df, bck_columns)

    logging.debug('Added columns after bucketisation: %s', len(df.columns) - total_columns)

    if converters is not None:
        logging.debug('Applying converters on columns')
        convert_columns(df, converters)

    return df


def make_compatible(cv, train):
    cols_to_add_train = set(cv.columns) - set(train)
    cols_to_add_cv = set(train.columns) - set(cv)

    for col in cols_to_add_cv:
        cv[col] = np.float32(0)

    for col in cols_to_add_train:
        train[col] = np.float32(0)


def __split_x_y(df, label_column):
    logging.debug('Splitting columns for x and y')
    y = df[label_column]
    df.drop(label_column, axis=1, inplace=True)
    x = df
    return x, y


def read_data(data_path, feature_names, label_column, bucketised_columns=None, converters_map=None, limit=None):
    logging.debug('Importing file %s', data_path)
    data = __read_bucketised_data(data_path, feature_names, limit, bucketised_columns, converters_map)

    return __split_x_y(data, label_column)
