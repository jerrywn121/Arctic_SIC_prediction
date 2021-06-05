import numpy as np
import pathlib
from scipy import interpolate
import torch.nn.functional as F
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
import datetime
import xarray as xr


def cal_time_length(start_time, end_time):
    """
    calculate the number of months between two dates
    """
    assert len(str(start_time)) == 6 & len(str(end_time)) == 6
    start_year = int(str(start_time)[:4])
    start_month = int(str(start_time)[4:])
    end_year = int(str(end_time)[:4])
    end_month = int(str(end_time)[4:])
    assert (start_month <= 12) & (start_month >= 1)
    assert (end_month <= 12) & (end_month >= 1)
    length = (12 - start_month + 1) + (end_year - start_year - 1) * 12 + end_month
    return length


def GenMonthList(start_month, end_month):
    """
    generate a list of months
    """
    months = []
    current = datetime.datetime.strptime(str(start_month) + '1', '%Y%m%d').date()
    end_month = datetime.datetime.strptime(str(end_month) + '1', '%Y%m%d').date()

    while current <= end_month:
        months.append(current.strftime('%Y%m'))
        current += relativedelta(months=1)

    return [int(x) for x in months]


def read_from_file(data_file):
    """
    read the data as image from the file
    """
    img = []
    for byte in pathlib.Path(data_file).read_bytes():
        img.append(byte)

    return np.array(img[300:]).reshape(448, 304)  # 448 rows and 304 columns


def read_process_data(txt, start_time, end_time):
    """
    read and process all the data
    Args:
        start_time/end_time: YYYYMM (200001)
        txt: the .txt file containing a list of file names to read as images
    Returns:
        imgs: the SIC data
        train_masks: regions to be trained are masked
    """
    assert len(str(start_time)) == 6 & len(str(end_time)) == 6
    with open(txt, 'r') as f:
        img_names = np.array(f.read().split())
        times = np.array([int(x[-21:-15]) for x in img_names])
        img_names = img_names[(times >= start_time) & (times <= end_time)]

    imgs = []
    train_masks = []
    for img_name in img_names:
        img = read_from_file(img_name)
        img, train_mask = post_process_data(img)
        imgs.append(img)
        train_masks.append(train_mask)

    assert len(imgs) == cal_time_length(start_time, end_time)
    return np.stack(imgs, axis=0), np.stack(train_masks, axis=0)


def write_netcdf(data_paths_file, start_time, end_time, out_path):
    """
    write the processed data into .nc file facilitate data reading next time
    Args:
        data_paths_file is the path to the text file containing all data file paths to be read for process,
        this file can be generated using the 'gen_data_text' script given that the data folder is aranged in
        an appropriate manner
    """
    imgs, train_masks = read_process_data(data_paths_file, start_time, end_time)
    ds = xr.Dataset({'imgs': (['time', 'x', 'y'], imgs), 'train_masks': (['time', 'x', 'y'], train_masks)},
                    coords={'time': GenMonthList(start_time, end_time), 'x': range(448), 'y': range(304)})
    ds.to_netcdf(out_path)
    return ds


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    """
    Args:
        input_gap=1: time gaps between two consecutive input frames
        input_length=12: the number of input frames
        pred_shift=26: the lead_time of the last target to be predicted
        pred_length=26: the number of frames to be predicted
        samples_gap: the gap between the starting time of two retrieved samples
    Returns:
        idx_inputs: indices pointing to the positions of input samples
        idx_targets: indices pointing to the positions of target samples
    """
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + pred_length), dtype=int)
    idx_inputs = ind[::samples_gap, :input_length]
    idx_targets = ind[::samples_gap, input_length:]
    return idx_inputs, idx_targets


def post_process_data(img):
    """
    deal with normalization, missing data, land masking and so on
    0 - 250 Sea ice concentration (fractional coverage scaled by 250)
    251 Circular mask used in the Arctic to cover the irregularly-shaped
    data gap around the pole (caused by the orbit inclination and instrument swath)
    the position of the circular mask may be different throughout the data
    252 Unused (found none in the data)
    253 Coastlines
    254 Superimposed land mask
    255 Missing data (found none in the data)
    """
    train_mask = (img<=250) & (img>=0)
    img[img==254] = -25
    img[img==253] = -25
    img = fill_missing_value(img, 251)
    assert not np.any(img>250)

    return img/250., train_mask


def fill_missing_value(data, value):
    """
    fill undefined missing value in the images with 2d nearest neighbourhood,
    applicable to 2d image data only
    Args:
        data: the SIC data to be filled in position data == value
        value: the value representing the missing value in the data,
    Returns:
        output with missing value positions filled by its neighbourhood
    """
    assert len(data.shape) == 2
    locs_non_na = (data != value).nonzero()
    locs_non_na = list(zip(*locs_non_na))
    locs_na = (data == value).nonzero()
    locs_na = list(zip(*locs_na))

    f = interpolate.NearestNDInterpolator(locs_non_na, data[data != value])
    data[data == value] = f(locs_na)

    return data


def unfold_StackOverChannel(img, kernel_size):
    """
    patch the image and stack individual patches along the channel
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, *, C*H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5
    if kernel_size[0] == 1 and kernel_size[1] == 1:
        return img

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor shape (N, *, C*k_h*k_w, n_h, n_w)
        output_size: (H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        note that the stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    if kernel_size[0] == 1 and kernel_size[1] == 1:
        return tensor
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded


class SIC_dataset(Dataset):
    def __init__(self, full_data_path, start_time, end_time, input_gap, input_length, pred_shift, pred_length, samples_gap, sie_mask_period=None):
        """
        Args:
            full_data_path: the path specifying where the processed data file is located
            start_time/end_time: used to specify the dataset (train/eval/test) period
            sie_mask_period: the time period used to find the grid cells where the sea ice has ever appeared,
                             this is used to prevent the model attending to open sea area during training,
                             and also used during evaluation to better evaluate model performance
        """
        super().__init__()

        self.start_time = start_time
        self.end_time = end_time
        with xr.open_dataset(full_data_path) as full_data:
            months = GenMonthList(start_time, end_time)
            self.train_masks = full_data.train_masks.sel(time=months).values
            data = full_data.imgs.sel(time=months).values
            if sie_mask_period is not None:
                self.train_masks = np.logical_and(self.train_masks, np.any(full_data.imgs.sel(time=GenMonthList(*sie_mask_period)).values > 0, axis=0, keepdims=True))

        idx_inputs, idx_targets = prepare_inputs_targets(data.shape[0], input_gap=input_gap, input_length=input_length,
                                                         pred_shift=pred_shift, pred_length=pred_length, samples_gap=samples_gap)

        self.train_masks = self.train_masks[idx_targets][:, :, None]
        self.inputs = data[idx_inputs][:, :, None]
        self.targets = data[idx_targets][:, :, None]
        self.months = np.array(months)[np.concatenate([idx_inputs, idx_targets], axis=1)]

    def GetDataShape(self):
        return {'train_masks': self.train_masks.shape,
                'inputs': self.inputs.shape,
                'targets': self.targets.shape}

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.train_masks[index]
