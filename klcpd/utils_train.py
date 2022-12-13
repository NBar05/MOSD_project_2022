import argparse

import math
import random
import numpy as np
import pandas as pd

from scipy.io import loadmat

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_auc_score

import pytorch_lightning as pl
from typing import List, Tuple
from tqdm import tqdm

from utils_models import NetD, NetG

# -------------------------------------------------------------------------------------- #
#                                       Datasets                                         #
# -------------------------------------------------------------------------------------- #

class SeriesDateset(Dataset):
    """

    Args for init:
    - Y: timeseries data
    - p_wnd_dim: past window size
    - f_wnd_dim: future window size

    """

    def __init__(self, Y: np.array, L: np.array, p_wnd_dim: int = 10, f_wnd_dim: int = 10):
        super().__init__()

        self.p_wnd_dim = p_wnd_dim
        self.f_wnd_dim = f_wnd_dim

        self.Y = Y
        self.T, self.D = Y.shape

        self.L = L

        self.Y_pad = self.pad_data()

    def pad_data(self):
        Y_pad = np.zeros((self.T + self.p_wnd_dim + self.f_wnd_dim, self.D))
        Y_pad[self.p_wnd_dim:-self.f_wnd_dim] = self.Y

        return np.float32(Y_pad)

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        start_past = idx
        end_past = start_future = idx + self.p_wnd_dim
        end_future = idx + self.p_wnd_dim + self.f_wnd_dim

        return {
            'X_p': self.Y_pad[start_past:end_past, :],
            'X_f': self.Y_pad[start_future:end_future, :],
            'Y': self.Y_pad[start_past:end_future, :],
            'L': self.L[idx],
        }


# -------------------------------------------------------------------------------------- #
#                                          Loss                                          #
# -------------------------------------------------------------------------------------- #

def median_heuristic(X, beta=0.5):
    max_n = min(10000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]


def batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var):
    device = X_p_enc.device
    # some constants, TODO ask Alex
    n_basis = 1024
    gumbel_lmd = 1e+6
    cnst = math.sqrt(1. / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, seq_len, nz = X_p_enc.size()

    # gumbel trick to get masking matrix to uniformly sample sigma
    def sample_gmm(W, batch_size):
        U = torch.FloatTensor(batch_size * n_samples, n_mixtures).uniform_()
        U = U.to(W.device)

        sigma_samples = F.softmax(U * gumbel_lmd, dim=1).matmul(sigma_var)

        W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
        W_gmm = W_gmm.view(batch_size, n_samples, nz)

        return W_gmm

    W = torch.FloatTensor(batch_size * n_samples, nz).normal_(0, 1)
    W = W.to(device)
    W.requires_grad = False
    W_gmm = sample_gmm(W, batch_size)  # batch_size x n_samples x nz
    W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()  # batch_size x nz x n_samples

    XW_p = torch.bmm(X_p_enc, W_gmm)  # batch_size x seq_len x n_samples
    XW_f = torch.bmm(X_f_enc, W_gmm)  # batch_size x seq_len x n_samples
    z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
    z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
    batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1)) ** 2, 1)

    return batch_mmd2_rff


def mmdLossD(X_f,
             Y_f,
             X_f_enc,  # real (initial)   subseq (future window)
             Y_f_enc,  # fake (generated) subseq (future window)
             X_p_enc,  # real (initial)   subseq (past window)
             X_f_dec,
             Y_f_dec,
             lambda_ae,
             lambda_real,
             sigma_var):

    # batchwise MMD2 loss between X_f and Y_f
    D_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)

    # batchwise MMD2 loss between X_p and X_f
    mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)

    # reconstruction loss
    real_L2_loss = torch.mean((X_f - X_f_dec) ** 2)
    fake_L2_loss = torch.mean((Y_f - Y_f_dec) ** 2)

    lossD = D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()

    return lossD.mean(), mmd2_real.mean()


# -------------------------------------------------------------------------------------- #
#                                        Models                                          #
# -------------------------------------------------------------------------------------- #



class KLCPD(pl.LightningModule):
    def __init__(
        self,
        args: dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int = 80
    ) -> None:
        super().__init__()

        self.args = args

        self.netD = NetD(self.args)
        self.netG = NetG(self.args)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.sigma_var = torch.FloatTensor(median_heuristic(self.train_dataset.Y, beta=.5))

        # to get predictions
        self.num_workers = num_workers


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        X_p, X_f = batch['X_p'], batch['X_f']

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)

        Y_pred = batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(X_p.device))

        return Y_pred


    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False
    ) -> None:

        # update generator every CRITIC_ITERS steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.args.CRITIC_ITERS == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()

        # update discriminator every step
        if optimizer_idx == 1:
            for p in self.netD.rnn_enc_layer.parameters():
                p.data.clamp_(-self.args.weight_clip, self.args.weight_clip)

            optimizer.step(closure=optimizer_closure)


    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> torch.Tensor:

        # optimize discriminator (netD)
        if optimizer_idx == 1:
            X_p, X_f = batch['X_p'], batch['X_f']

            # real data
            X_p_enc, X_p_dec = self.netD(X_p)
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            Y_f = self.netG(X_p, X_f)

            Y_f_enc, Y_f_dec = self.netD(Y_f)

            lossD, mmd2_real = mmdLossD(X_f, Y_f, X_f_enc, Y_f_enc, X_p_enc, X_f_dec, Y_f_dec,
                                        self.args.lambda_ae, self.args.lambda_real, self.sigma_var.to(self.device))
            lossD = (-1) * lossD
            self.log("train_loss_D", lossD, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)

            return lossD

        # optimize generator (netG)
        if optimizer_idx == 0:
            X_p, X_f = batch['X_p'], batch['X_f']

            # real data
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            Y_f = self.netG(X_p, X_f)
            Y_f_enc, Y_f_dec = self.netD(Y_f)

            # batchwise MMD2 loss between X_f and Y_f
            G_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var.to(self.device))

            lossG = G_mmd2.mean()
            self.log("train_loss_G", lossG, prog_bar=True)

            return lossG


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X_p, X_f = batch['X_p'], batch['X_f']

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)

        val_mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(self.device))

        lossD = val_mmd2_real.mean()
        self.log('val_mmd2_real_D', lossD, prog_bar=True)

        return lossD


    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:

        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        return optimizerG, optimizerD


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=self.args.batch_size, shuffle=False, num_workers=self.num_workers)


# -------------------------------------------------------------------------------------- #
#                                       Evaluation                                       #
# -------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

    parser.add_argument('--path_to_data_folder', type=str, default='klcpd_code/data/', help='path to data folder')
    parser.add_argument('--path_to_dataset', type=str, default='hasc/hasc-1.mat', help='path to dataset')
    parser.add_argument('--D', type=int, default=3, help='number of RNN features')

    parser.add_argument('--trn_ratio', type=float, default=0.6, help='how much data used for training')
    parser.add_argument('--val_ratio', type=float, default=0.8, help='how much data used for validation')

    parser.add_argument('--wnd_dim', type=int, default=5, help='window size (past and future)')

    # RNN hyperparemters
    parser.add_argument('--RNN_hid_dim', type=int, default=10, help='number of RNN hidden units')
    parser.add_argument('--num_layers', type=int, default=10, help='number of RNN layers')

    # optimization
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--optim', type=str, default='adam', help='sgd|rmsprop|adam for optimization method')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 regularization)')

    # GAN
    parser.add_argument('--CRITIC_ITERS', type=int, default=5, help='number of updates for critic per generator')
    parser.add_argument('--weight_clip', type=float, default=.1, help='weight clipping for crtic')
    parser.add_argument('--lambda_ae', type=float, default=0.001, help='coefficient for the reconstruction loss')
    parser.add_argument('--lambda_real', type=float, default=0.1, help='coefficient for the real MMD2 loss')

    args = parser.parse_args()

    scores = []
    for seed in range(0, 5, 1):
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        path_to_data_folder = args.path_to_data_folder # 'klcpd_code/data/'
        path_to_dataset = args.path_to_dataset # 'beedance/beedance-1.mat', 'fishkiller/fishkiller.mat'

        data = loadmat(path_to_data_folder + path_to_dataset)
        T = len(data['Y'])

        train_slice = slice(0, int(np.ceil(args.trn_ratio * T)))
        valid_slice = slice(int(np.ceil(args.trn_ratio * T)), int(np.ceil(args.val_ratio * T)))
        test_slice  = slice(int(np.ceil(args.val_ratio * T)), None)

        dataset_train = SeriesDateset(data['Y'][train_slice], data['L'][train_slice], p_wnd_dim=args.wnd_dim, f_wnd_dim=args.wnd_dim)
        dataset_valid = SeriesDateset(data['Y'][valid_slice], data['L'][valid_slice], p_wnd_dim=args.wnd_dim, f_wnd_dim=args.wnd_dim)
        dataset_test  = SeriesDateset(data['Y'][test_slice] , data['L'][test_slice] , p_wnd_dim=args.wnd_dim, f_wnd_dim=args.wnd_dim)

        model = KLCPD(args, train_dataset=dataset_train, valid_dataset=dataset_valid, test_dataset=dataset_test, num_workers=2)

        trainer = pl.Trainer(accelerator='gpu', devices=1, 
                             max_epochs=5, check_val_every_n_epoch=5, gradient_clip_val=args.weight_clip)
        trainer.fit(model)

        y_pred, y_true = [], []
        test_dataloader = model.test_dataloader()
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            y_pred.extend(model.forward(batch).detach().numpy())
            y_true.extend(batch['L'].detach().numpy().flatten())

        score = roc_auc_score(y_true, y_pred)

        print()
        print(path_to_data_folder + path_to_dataset, round(score, 5))
        
        scores.append(score)

    with open(f"{path_to_data_folder.replace('/', '_') + path_to_dataset.replace('/', '_')}_{args.wnd_dim}.txt", 'w') as f:
        f.write(str(scores))
