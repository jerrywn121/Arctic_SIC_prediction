#!/bin/bash

data_path=./data/full_sic.nc
python3 test.py -ts 201712 -te 202012 -o ./test_result --full_data_path ${data_path}
