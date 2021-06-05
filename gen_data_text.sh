#!/bin/bash

DATA_DIR=./data
TEXTFILE=./data/data.txt

for year in `ls $DATA_DIR`
    do
        for datafile in `ls ${DATA_DIR}/${year}`
            do
                echo $datafile
                echo ${DATA_DIR}/${year}/$datafile >> $TEXTFILE
        done
done
