import pandas as pd
import tensorflow as tf
import tempfile
import numpy as np
train_file = open("./data/train.csv", "r")
test_file = open("./data/test.csv", "r")

COLUMNS = ["WORK_CENTRE_CD", "ARTICLE_ID", "PRODUCT_CD", "RECEIVER_SUBURB", "THOROUGHFARE_TYPE_CODE", "SIDE",
           "RECEIVER_DPID", "ADDRESS_CLUSTER", "SCAN_EVENT_CD", "DEVICE_USER_ID", "SCAN_SOURCE_DEVICE", "USER_ROLE",
           "CONTRACT_ID", "EVENT_TIMESTAMP", "DELIVERY_DATE", "DELIVERY_WEEKDAY", "DELIVERY_TIME", "ACCEPT_TIME",
           "NUMERIC_TIME"]

DTYPE = { "PRODUCT_CD": str,
          "RECEIVER_SUBURB": str,
          "THOROUGHFARE_TYPE_CODE": str,
          "SIDE": str,
          "RECEIVER_DPID": str,
          "ADDRESS_CLUSTER": str,
          "DEVICE_USER_ID": str,
          "USER_ROLE": str,
          "CONTRACT_ID": str,
          "DELIVERY_WEEKDAY": str,
          "NUMERIC_TIME": str,
          "ACCEPT_TIME": str,
          }
df_train = pd.read_csv(train_file, names=COLUMNS, dtype=DTYPE, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS,dtype=DTYPE, skipinitialspace=True, skiprows=1)

RESULT_COLUMN = "RESULT"
# df_train["ACCEPT_TIME_NUMERIC_TIME"] = (
#     df_train["ACCEPT_TIME"].apply(lambda x: float(x.split(":")[1]) / 60.0 + float(x.split(":")[0]))).astype(float)

df_train["RESULT"] = (
    df_train["NUMERIC_TIME"].apply(lambda x: str(int(float(x)/2)))).astype(str)


# df_test["ACCEPT_TIME_NUMERIC_TIME"] = (
#     df_test["ACCEPT_TIME"].apply(lambda x: float(x.split(":")[1]) / 60.0 + float(x.split(":")[0]))).astype(float)

df_test["RESULT"] = (
    df_test["NUMERIC_TIME"].apply(lambda x: str(int(float(x)/2)))).astype(str)




CATEGORICAL_COLUMNS = [
    "PRODUCT_CD", "RECEIVER_SUBURB", "THOROUGHFARE_TYPE_CODE", "SIDE",
    "RECEIVER_DPID", "ADDRESS_CLUSTER", "DEVICE_USER_ID", "USER_ROLE", "CONTRACT_ID", "DELIVERY_WEEKDAY"]

product_code = tf.contrib.layers.sparse_column_with_hash_bucket("PRODUCT_CD", hash_bucket_size=1000)
receiver_suburb = tf.contrib.layers.sparse_column_with_hash_bucket("RECEIVER_SUBURB", hash_bucket_size=1000)
thoroughfare_type_code = tf.contrib.layers.sparse_column_with_hash_bucket("THOROUGHFARE_TYPE_CODE",
                                                                          hash_bucket_size=1000)
side = tf.contrib.layers.sparse_column_with_hash_bucket("SIDE", hash_bucket_size=3)
receiver_dpid = tf.contrib.layers.sparse_column_with_hash_bucket("RECEIVER_DPID", hash_bucket_size=1000)
address_cluster = tf.contrib.layers.sparse_column_with_hash_bucket("ADDRESS_CLUSTER", hash_bucket_size=1000)
device_user_id = tf.contrib.layers.sparse_column_with_hash_bucket("DEVICE_USER_ID", hash_bucket_size=1000)
user_role = tf.contrib.layers.sparse_column_with_hash_bucket("USER_ROLE", hash_bucket_size=1000)
contract_id = tf.contrib.layers.sparse_column_with_hash_bucket("CONTRACT_ID", hash_bucket_size=1000)
delivery_weekday = tf.contrib.layers.sparse_column_with_hash_bucket("DELIVERY_WEEKDAY", hash_bucket_size=8)

# receiver_suburb_x_weekday = tf.contrib.layers.crossed_column(
#     [receiver_suburb, delivery_weekday], hash_bucket_size=int(1e6))

# CONTINUOUS_COLUMNS = ["ACCEPT_TIME_NUMERIC_TIME"]


# accept_numeric_time = tf.contrib.layers.real_valued_column("ACCEPT_TIME_NUMERIC_TIME")
#

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {}
    # continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
    #                    for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[RESULT_COLUMN].values, shape=[df[RESULT_COLUMN].size, 1])
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
    product_code,
    receiver_suburb,
    thoroughfare_type_code,
    side,
    receiver_dpid,
    address_cluster,
    device_user_id,
    user_role,
    contract_id,
    delivery_weekday,
],
    model_dir=model_dir)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])