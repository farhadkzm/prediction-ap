import pandas as pd

train_file = open("./data/newtown_train.csv", "r")
test_file = open("./data/newtown_test.csv", "r")

COLUMNS = ["WORK_CENTRE_CD", "ARTICLE_ID", "PRODUCT_CD", "RECEIVER_SUBURB", "THOROUGHFARE_TYPE_CODE", "SIDE",
           "RECEIVER_DPID", "ADDRESS_CLUSTER", "SCAN_EVENT_CD", "DEVICE_USER_ID", "SCAN_SOURCE_DEVICE", "USER_ROLE",
           "CONTRACT_ID", "EVENT_TIMESTAMP", "DELIVERY_DATE", "DELIVERY_WEEKDAY", "DELIVERY_TIME", "ACCEPT_TIME",
           "NUMERIC_TIME"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

df_train["ACCEPT_TIME_NUMERIC_TIME"] = (df_train["ACCEPT_TIME"].apply(lambda x: float(x.split(":")[1])/60.0 + float(x.split(":")[0]))).astype(float)


CATEGORICAL_COLUMNS = [
    "PRODUCT_CD", "RECEIVER_SUBURB","THOROUGHFARE_TYPE_CODE", "SIDE",
    "RECEIVER_DPID", "ADDRESS_CLUSTER", "DEVICE_USER_ID", "USER_ROLE","CONTRACT_ID","DELIVERY_WEEKDAY"]
CONTINUOUS_COLUMNS = ["ACCEPT_TIME_NUMERIC_TIME"]
