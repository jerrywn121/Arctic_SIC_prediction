from trainer import Trainer
import pickle
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from utils import SIC_dataset
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--start_time', action='store', help="six-digits number", type=int, required=True)
    parser.add_argument('-te', '--end_time', action='store', help="six-digits number", type=int, required=True)
    parser.add_argument('-o', '--output_dir', action='store', default=None,
                        help='specify where to save the prediction output', type=Path, required=True)
    parser.add_argument('--full_data_path', action='store', default=None,
                        help='.nc file storing the read data', type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    log_file = args.output_dir / Path('test.log')
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
    logger = logging.getLogger()

    with open('config_train.pkl', 'rb') as config_test:
        configs = pickle.load(config_test)
    logger.info(configs.__dict__)
    logger.info(f'Arguments:\n\
            start time: {args.start_time}\n\
            end time: {args.end_time}\n\
            output_dir: {args.output_dir}\n\
            full_data_path: {args.full_data_path}\n')
    
    # load the dataset and dataloader
    logger.info(f'\nloading test dataset from {args.start_time} to {args.end_time}')
    dataset_test = SIC_dataset(args.full_data_path, args.start_time, args.end_time,
                               configs.input_gap, configs.input_length, configs.pred_shift, configs.output_length,
                               samples_gap=1, sie_mask_period=configs.sie_mask_period)
    logger.info('loaded test set data is of shape:')
    logger.info(dataset_test.GetDataShape())
    logger.info('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)

    # load and test model
    logger.info('testing...')
    tester = Trainer(configs, np.load('land_mask.npy'))
    tester.network.load_state_dict(torch.load('checkpoint.chk')['net'])
    rmse, mae, acc, sic_pred = tester.infer(dataset_test, dataloader_test)
    logger.info("rmse: {:.5f}, mae: {:.5f}, acc: {:.5f}\n".format(rmse, mae, acc))

    # save y_true, y_pred and masks
    logger.info(f'saving outout to {args.output_dir}')
    np.save(args.output_dir / 'sic_pred.npy', sic_pred.cpu().numpy())
    np.save(args.output_dir / 'inputs.npy', dataset_test.inputs)
    np.save(args.output_dir / 'targets.npy', dataset_test.targets)
    np.save(args.output_dir / 'train_masks.npy', dataset_test.train_masks)
    np.save(args.output_dir / 'months.npy', dataset_test.months)
    logger.info('Done')


if __name__ == '__main__':
    main()
