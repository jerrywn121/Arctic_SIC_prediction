# Arctic Sea Ice Concentration prediction
Models for Predicting Sea Ice Concentration (SIC) in the Arctic.
The study is based on the fact that the literature generally fail to model the spatiotemporal evolution of the sea ice concentration


## Models
We adopt the ConvLSTM for SIC prediction by taking the SIC as the input only, with reference to two baselines, i.e., LSTM and Damped Anomaly Persistence. See a detailed comparison and evluation of these models in the model_result_analysis.ipynb
- ConvLSTM (Shi et al., 2015)
- LSTM\
In practice, the LSTM is realized by training the ConvLSTM model with kernel size set to 1
- Damped Anomaly Persistence


## How to Use
- **Data**\
We suggest the following way of handling the data, but there can be a more convinient way.
  1. The monthly SIC data used in this study can be downloaded from National Snow and Ice Data Center (NSIDC) https://nsidc.org/data/nsidc-0051, which also contains a detailed description of the dataset and user guide. Download the data and arange them as **data/year/monthly_file**
  2. generate the .txt file containing the path to the all data files
  ```
  ./gen_data_text.sh
  ```
  3. read and store the processed data to facilitate data reading next time, by calling the write_netcdf function and pass the output file name in the previous step as the argument, e.g.,
  ```
  from utils import write_netcdf
  write_netcdf("./data/data.txt", start_time, end_time, "./data/full_sic.nc")
  ```

- **Train**\
change relevant parameters in the config.py file, put all the scripts under the same folder and run
```
python train.py
```
the training process will be printed and one could also choose to direct these info to other logging files

- **test**\
specifying the testing period and output directory and run
```
./test.sh
```

- **Compare Models**\
We also provide details for evaluating the models in model_result_analysis.ipynb, containing different metrics and plotting.


