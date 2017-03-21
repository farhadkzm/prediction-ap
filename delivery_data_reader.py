import data_reader
import pandas as pd
import logging

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


def __convert_time_to_float(value):
    digits = value.split(':')
    return float(digits[0]) + float(digits[1]) * 1 / 60


converters_map = {'ACCEPT_NUMERIC': ('ACCEPT_TIME', __convert_time_to_float)}


def __read_data(data_path, limit=None):
    return data_reader.read_data(data_path, feature_names, 'NUMERIC_TIME', bucketised_columns, converters_map,
                                 limit=limit)


def initialise_data(train_path, test_path, out_dir_path, train_size=None, test_size=None, write=False):
    # reading data
    train_x, train_y = __read_data(train_path, train_size)
    testcv_x, testcv_y = __read_data(test_path, test_size)

    data_reader.make_compatible(testcv_x, train_x)

    # splitting data into different sets
    ratios = [.5]
    test_x, cv_x = data_reader.split_data(testcv_x, ratios)
    test_y, cv_y = data_reader.split_data(testcv_y, ratios)

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
