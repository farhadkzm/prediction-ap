Extracting Newtown data from workcenter 266095:

`cat data.csv | grep ^\"266095\" | grep \"NEWTOWN\" | grep -v ',,' | shuf | sed 's/\"//g' > newtown_data.csv`


Extracting training data and test data (no. of lines is 17303)

`head -n 14303 newtown_cleansed.csv >  newtown_train.csv`

`tail -n 3000 newtown_cleansed.csv >  newtown_test.csv`

