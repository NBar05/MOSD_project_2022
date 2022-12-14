import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 dropout_rate: float = 0, 
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False, 
                 training: bool = True):
        """Defines the residual block for the WaveNet TCN
        Args:
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        # for causal padding
        self.padding = (self.kernel_size - 1) * self.dilation_rate
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        
        self.training = training

        super(ResidualBlock, self).__init__()
        
        self.conv_1 = nn.Conv1d(in_channels, self.nb_filters, self.kernel_size, 
                                padding=0, dilation=self.dilation_rate)        
        if self.use_weight_norm:
            weight_norm(self.conv_1) 
        self.bn_1 = nn.BatchNorm1d(self.nb_filters)
        self.ln_1 = nn.LayerNorm(self.nb_filters)              
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv1d(self.nb_filters, self.nb_filters, self.kernel_size, 
                                padding=0, dilation=self.dilation_rate)        
        if self.use_weight_norm:
            weight_norm(self.conv_1)    
        self.bn_2 = nn.BatchNorm1d(self.nb_filters)
        self.ln_2 = nn.LayerNorm(self.nb_filters)              
        self.relu_2 = nn.ReLU()        
        
        self.conv_block = nn.Sequential()
        self.downsample = nn.Conv1d(in_channels, self.nb_filters, kernel_size=1) if in_channels != self.nb_filters else nn.Identity()
        
        self.relu = nn.ReLU()  
                
        self.init_weights()
        
        
    def init_weights(self):
        # in the realization, they use random normal initialization
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_1.bias)            
        
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_2.bias)            
        
        if isinstance(self.downsample, nn.Conv1d):         
            torch.nn.init.normal_(self.downsample.weight, mean=0, std=0.05)
            torch.nn.init.zeros_(self.downsample.bias)                    
            
    def forward(self, inp):
        # inp batch, channels, time
        ######################
        # do causal padding        
        out = F.pad(inp, (self.padding, 0))
        out = self.conv_1(out)
        
        if self.use_batch_norm:
            out = self.bn_1(out)
        elif self.use_layer_norm:
            out = self.ln_1(out)        
        out = self.relu_1(out)
        
        # spatial dropout
        out = out.permute(0, 2, 1)   # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)        
        out = out.permute(0, 2, 1)   # back to [batch, channels, time]    
        
        #######################
        # do causal padding
        out = F.pad(out, (self.padding, 0))
        out = self.conv_2(out)
        if self.use_batch_norm:
            out = self.bn_2(out)
        elif self.use_layer_norm:
            out = self.ln_2(out)
        out = self.relu_2(out)            
        out = self.relu_2(out)    
        # spatial dropout
        # out batch, channels, time 
        
        out = out.permute(0, 2, 1)   # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)
        out = out.permute(0, 2, 1)   # back to [batch, channels, time]            
        
        #######################        
        skip_out = self.downsample(inp)
        #######################
        res = self.relu(out + skip_out)
        return res, skip_out
    
# only causal padding
# only return sequence = True
    
class TCN(nn.Module):        
    def __init__(self,
                 in_channels=1,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 use_skip_connections=True,
                 dropout_rate=0.0, 
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False, 
                 use_weight_norm: bool = False):

        super(TCN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.in_channels = in_channels
        
        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')        
        
        self.residual_blocks = []        
        res_block_filters = 0
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                in_channels = self.in_channels if i + s == 0 else res_block_filters                
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(in_channels=in_channels, 
                                                          dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          dropout_rate=self.dropout_rate, 
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm))

        
        self.residual_blocks = nn.ModuleList(self.residual_blocks)
                                            
    def forward(self, inp):
        out = inp
        for layer in self.residual_blocks:
            out, skip_out = layer(out)
        if self.use_skip_connections:
            out = out + skip_out
        return out

########################### model #########################################
class Encoder(nn.Module):
    def __init__(self, c_in=1, nb_filters=64, kernel_size=4, 
                 dilations=[1,4,16], nb_stacks=2, n_steps=50, code_size=10, seq_len=120):       
        super(Encoder, self).__init__()        
        
        self.tcn_layer = TCN(in_channels=c_in, nb_filters=nb_filters, 
                             nb_stacks=nb_stacks, dilations=dilations, use_skip_connections=True, dropout_rate=0)
        
        self.fc1 = nn.Linear(nb_filters * seq_len, 2 * n_steps)  
        self.fc2 = nn.Linear(2 * n_steps, n_steps)    
        self.output_layer = nn.Linear(n_steps, code_size)           
        self.relu = nn.ReLU()
        print("init done")
        
    def forward(self, x):
        out = x
        
        #print("out.shape after initial x", out.shape)
        
        if len(out.shape) == 2:
            out = out.unsqueeze(1)
            
        #print("out.shape after unsqueeze(1) if len(out.shape) == 2", out.shape)
        
        out = self.tcn_layer(out)   
        
        #print("out.shape after self.tcn_layer(out)", out.shape)
        
        out = out.flatten(1, 2)     
        
        #print("out.shape after out.flatten(1, 2)", out.shape)
        
        out = self.relu(self.fc1(out)) 
        
        #print("out.shape after self.relu(self.fc1(out))", out.shape)
        
        out = self.relu(self.fc2(out))
        
        #print("out.shape after self.relu(self.fc2(out))", out.shape)
        
        out = self.output_layer(out)
        
        #print("out.shape after self.output_layer(out)", out.shape)
        
        return out
    
########################### loss #########################################
def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    v = cos(x.unsqueeze(1), y.unsqueeze(0))
    return v    

def nce_loss_fn(history, future, similarity, temperature=0.5):
    try:
        device = history.device
    except:
        device = 'cpu'
        
    criterion = torch.nn.BCEWithLogitsLoss()
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = torch.exp(torch.diag(sim) / temperature)

    tri_mask = torch.ones((N, N), dtype=bool)
    tri_mask[np.diag_indices(N)] = False
    
    neg = sim[tri_mask].reshape(N, N - 1)    
    all_sim = torch.exp(sim / temperature)
    
    logits = torch.divide(torch.sum(pos_sim), torch.sum(all_sim, axis=1))
        
    lbl = torch.ones(history.shape[0]).to(device)
    # categorical cross entropy
    loss = criterion(logits, lbl)    
    # loss = K.sum(logits)
    # divide by the size of batch
    #loss = loss / lbl.shape[0]
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.diag(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg


######################### preprocessing ###########################

# def _history_future_separation(mbatch, win):    
#     x = mbatch[:,1:win+1]
#     y = mbatch[:,-win:]
#     return x, y


################## PL wrapper ###############################################
class TSCP_model(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,     
        train_dataset: Dataset, 
        test_dataset: Dataset, 
        batch_size: int = 64,        
        num_workers: int = 2,        
        temperature: float = 0.1, 
        lr: float = 1e-4,
        decay_steps: int = 1000, 
        window_1: int = 100,
        window_2: int = 100
    ) -> None:
        super().__init__()
                    
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.batch_size = batch_size
        self.num_workers = num_workers        
        
        self.temperature = temperature
        
        self.lr = lr
        self.decay_steps = decay_steps   
        
        self.window = window_1
        self.window_1 = window_1
        self.window_2 = window_2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        
        # print("batch", batch)
        # print("batch.shape", batch.shape)
        
        
        # print("batch[:,:self.window]", batch[:,:self.window])
        # print("batch[:,:self.window].shape", batch[:,:self.window].shape)

        # print("batch[:,self.window:]", batch[:,self.window:])
        # print("batch[:,self.window:].shape", batch[:,self.window:].shape)
        
        history, future = batch[:, :self.window], batch[:, self.window:]   
        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float()) 

        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)

        train_loss, pos_sim, neg_sim = nce_loss_fn(history_emb, future_emb, similarity=_cosine_simililarity_dim2, 
                                                   temperature=self.temperature)

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)   
        self.log("pos_sim", pos_sim, prog_bar=False, on_epoch=True)           
        self.log("neg_sim", neg_sim, prog_bar=False, on_epoch=True)        

        return train_loss
        
    def validation_step(self, batch, batch_idx):
        
        # print("batch", batch)
        # print("batch.shape", batch.shape)
        
        
        # print("batch[:,:self.window]", batch[:,:self.window])
        # print("batch[:,:self.window].shape", batch[:,:self.window].shape)

        # print("batch[:,self.window:]", batch[:,self.window:])
        # print("batch[:,self.window:].shape", batch[:,self.window:].shape)
        
        history, future = batch[:,:self.window], batch[:,self.window:]
                
        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float()) 
                
        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)
        

        val_loss, pos_sim, neg_sim = nce_loss_fn(history_emb, future_emb, similarity=_cosine_simililarity_dim2, 
                                                 temperature=self.temperature)

        self.log("val_loss", val_loss, prog_bar=False)     
        self.log("val_pos_sim", pos_sim, prog_bar=False)        
        self.log("val_neg_sim", neg_sim, prog_bar=False)        

        return val_loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=self.decay_steps)
        return opt

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
    
def _cosine_simililarity_dim1(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    v = cos(x, y)
    return v    
