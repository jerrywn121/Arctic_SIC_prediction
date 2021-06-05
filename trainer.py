from model import IceNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import math


class Trainer:
    def __init__(self, configs, land_mask):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(10)
        self.network = IceNet(configs, land_mask).to(configs.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=1, cooldown=1, verbose=True, min_lr=0.0001)

    def rmse_func(self, y_pred, y_true, mask):
        rmse = torch.sum((y_pred - y_true)**2*mask, dim=[1, 2, 3, 4]) / mask.sum(dim=[1, 2, 3, 4])
        rmse = rmse.sqrt().mean()
        return rmse

    def mae_func(self, y_pred, y_true, mask):
        mae = torch.abs(y_pred - y_true) * mask
        mae = mae.sum(dim=[1, 2, 3, 4]) / mask.sum(dim=[1, 2, 3, 4])
        return mae.mean()

    def acc_func(self, y_pred, y_true, mask):
        pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, *)
        true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, *)
        ExEy = (pred * true * mask).mean()
        Exy = torch.mean(pred**2*mask) * torch.mean(true**2*mask)
        cor = ExEy / Exy.sqrt()
        return cor

    def train_once(self, inputs, targets, train_mask):
        sic_pred = self.network(inputs.float().to(self.device))
        self.optimizer.zero_grad()
        rmse = self.rmse_func(sic_pred, targets.float().to(self.device), train_mask.to(self.device))
        mae = self.mae_func(sic_pred, targets.float().to(self.device), train_mask.to(self.device))

        (rmse + mae).backward()
        if self.configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.configs.clipping_threshold)
        self.optimizer.step()
        return rmse.item(), mae.item(), sic_pred

    def test(self, dataloader_test):
        sic_pred = []
        with torch.no_grad():
            for inputs, targets, _ in dataloader_test:
                pred = self.network(inputs.float().to(self.device))
                sic_pred.append(pred)

        return torch.cat(sic_pred, dim=0)

    def infer(self, dataset, dataloader):
        # provide information about loss_func and score for a eval/test set
        self.network.eval()
        with torch.no_grad():
            sic_pred = self.test(dataloader)
            sic_true = torch.from_numpy(dataset.targets).float().to(self.device)
            train_masks = torch.from_numpy(dataset.train_masks).to(self.device)
            rmse = self.rmse_func(sic_pred, sic_true, train_masks).item()
            mae = self.mae_func(sic_pred, sic_true, train_masks).item()
            acc = self.acc_func(sic_pred, sic_true, train_masks).item()
        return rmse, mae, acc, sic_pred

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(21)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()

            for j, (inputs, targets, train_mask) in enumerate(dataloader_train):
                rmse, mae, sic_pred = self.train_once(inputs, targets, train_mask)
                if j % self.configs.display_interval == 0:
                    print('batch training loss: rmse: {:.4f}, mae: {:.4f}'.format(rmse, mae))

            # evaluation
            rmse_eval, mae_eval, acc_eval, _ = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            print('epoch eval loss: rmse: {:.4f}, mae: {:.4f}, acc: {:.3f}'.format(rmse_eval, mae_eval, acc_eval))
            loss_eval = rmse_eval + mae_eval
            self.lr_scheduler.step(loss_eval)
            if loss_eval >= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model(chk_path)
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
