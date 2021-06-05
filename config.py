import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.device = torch.device('cuda:0')
configs.batch_size_test = 20
configs.batch_size = 1
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 50
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 5
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# patch
configs.patch_size = (2, 2)
configs.img_size = (448, 304)

# data related
configs.input_dim = 1
configs.output_dim = 1

configs.input_length = 13
configs.output_length = 1

configs.input_gap = 1
configs.pred_shift = 1

configs.train_period = (197811, 201512)
configs.eval_period = (201412, 201812)
configs.sie_mask_period = configs.train_period

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (96, 96, 96, 96)

configs.full_data_path = './data/full_sic.nc'
