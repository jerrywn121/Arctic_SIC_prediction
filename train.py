import torch
from config import configs
from trainer import Trainer
from utils import SIC_dataset
import numpy as np


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print(configs.__dict__)

    start_train, end_train = configs.train_period
    start_eval, end_eval = configs.eval_period
    input_gap = configs.input_gap
    input_length = configs.input_length
    pred_shift = configs.pred_shift
    output_length = configs.output_length

    print(f'loading train dataset from {start_train} to {end_train}')
    dataset_train = SIC_dataset(configs.full_data_path, start_train, end_train,
                                input_gap, input_length, pred_shift, output_length,
                                samples_gap=1, sie_mask_period=configs.sie_mask_period)
    print(dataset_train.GetDataShape())
    print(dataset_train.months[0])
    print(dataset_train.months[-1])

    print(f'loading eval dataset from {start_eval} to {end_eval}')
    dataset_eval = SIC_dataset(configs.full_data_path, start_eval, end_eval,
                               input_gap, input_length, pred_shift, output_length,
                               samples_gap=1, sie_mask_period=configs.sie_mask_period)
    print(dataset_eval.GetDataShape())
    print(dataset_eval.months[0])
    print(dataset_eval.months[-1])

    trainer = Trainer(configs, np.load('land_mask.npy'))
    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')
    print('\n######training finished!########\n')
