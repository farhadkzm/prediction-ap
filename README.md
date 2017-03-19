Data ETL practices:
* include headers as the first row of the file
* define a hash based on specific values in rows and make sure it's unique (df.groupby('HASH').nth(0) will get the first row in repetitive rows)
* data cleansing (dropping NA, etc.) should happen before reading data by trainer
* as bucketisation happens, it'd better to have one file
* if there is a logical category in data, for test/cv grab .4 of each category(not randomly on whole data)

Preparing raw data:
* keep only workcenter 266095
* find suburbs where their number of deliveries are more than 1000
* define hash column based on ARTICLE_ID and EVENT_TIMESTAMP
* keep only one row for duplicate hashes
* group data by RECEIVER_SUBURB and split to .6 and .4 portions for train and test respectively


