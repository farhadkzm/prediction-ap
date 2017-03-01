Extracting Newtown data from workcenter 266095:

`cat data.csv | grep ^\"266095\" | grep \"NEWTOWN\" > newtown_data.csv`

Randomising the lines in the file

`cat newtown_data.csv | shuf  > newtown_data_randomised.csv`

Removing the double quotes 

`sed 's/\"//g' newtown_data_randomised.csv > newtown_cleansed.csv`

Extracting training data and test data (no. of lines is 17303)

`head -n 14303 newtown_cleansed.csv >  newtown_train.csv`

`tail -n 3000 newtown_cleansed.csv >  newtown_test.csv`

