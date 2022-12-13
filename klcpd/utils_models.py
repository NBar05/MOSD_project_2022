import os
import random

import math
import numpy as np

import torch
from torch import nn


class NetG(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.rnn_enc_layer = nn.GRU(args.D, args.RNN_hid_dim, num_layers=args.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(args.D, args.RNN_hid_dim, num_layers=args.num_layers, batch_first=True)
        self.fc_layer = nn.Linear(args.RNN_hid_dim, args.D)

    # X_p:   batch_size x wnd_dim x var_dim (Encoder input)
    # X_f:   batch_size x wnd_dim x var_dim (Decoder input)
    # h_t:   1 x batch_size x RNN_hid_dim
    # noise: 1 x batch_size x RNN_hid_dim

    def forward(self, X_p, X_f):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)

        noise = torch.FloatTensor(1, len(X_p), self.args.RNN_hid_dim).normal_(0, 1)
        noise.requires_grad = False
        noise = noise.to(X_p.device)

        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0.0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.rnn_enc_layer = nn.GRU(args.D, args.RNN_hid_dim, num_layers=args.num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(args.RNN_hid_dim, args.D, num_layers=args.num_layers, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec
