# TO DO : import from flows configs.
#from src.configs.flows import *
#from src.configs.ais import AISConfig1

import torch
from torch import nn


class FCEncoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args

    self.layers =  nn.ModuleList()
    self.layers.append(nn.Linear(args.input_dim, args.hidden_dim))
    for i in range(args.num_deterministic_layers):
      self.layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
      self.layers.append(nn.ReLU())

    self.layers.append(nn.Linear(args.hidden_dim, args.latent_dim * 2))


  def forward(self, x):
    h = x
    for layer in self.layers:
      h = layer(h)
    return h


class FCDecoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args

    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(args.latent_dim, args.hidden_dim))
    for i in range(args.num_deterministic_layers):
      self.layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
      self.layers.append(nn.ReLU())

    self.layers.append(nn.Linear(args.hidden_dim, args.input_dim))
    self.layers.append(nn.Sigmoid())

   

  def forward(self, x):
    h = x
    for layer in self.layers:
      h = layer(h)
    return h



class ConvEncoder(nn.Module):
  def __init__(self, D, args):
    super().__init__()
    self.layers = nn.ModuleList()
    self.args = args
    if args.encoder_architecture == 'conv':
      self.layers.append(Reshape([1,28,28]))
      self.layers.append(nn.Conv2d(1, 32, 4, 2, 1))
      #self.layers.append(nn.BatchNorm2d(num_features=32))
      self.layers.append(nn.ReLU())
      self.layers.append(nn.Conv2d( 32, 32, 4, 2, 1))
      #self.layers.append(nn.BatchNorm2d(num_features=32))
      self.layers.append(nn.ReLU())
      self.layers.append(nn.Conv2d(32, 64, 3, 2, 0))
      #self.layers.append(nn.BatchNorm2d(num_features=64))
      self.layers.append(nn.ReLU())
      self.layers.append(nn.Conv2d( 64, 256, 3, 1, 0))
      #self.layers.append(nn.BatchNorm2d(num_features=256))
      self.layers.append(nn.ReLU())
      self.layers.append(Flatten())
      self.mu = nn.Linear(256, args.latent_dim)
      self.var = nn.Linear(256, args.latent_dim)
      # {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
      # {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
      # {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 2, "padding": 0, "activation": "relu", "batch_norm": true},
      # {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 0, "activation": "relu", "batch_norm": true},
      # {"type": "flatten"},
      # {"type": "gaussian", "dim": 128}


    elif args.encoder_architecture is not None:
      raise NotImplementedError
    else:
      for i in range(args.num_deterministic_layers+1):
        if i < args.num_deterministic_layers:
          self.layers.append(nn.Linear(D if i == 0 else args.hidden_dim, \
                        args.hidden_dim))
                        
          self.layers.append(args.non_linearity)
        else:
          self.mu = nn.Linear(args.hidden_dim, args.latent_dim).cuda()
          self.var = nn.Linear(args.hidden_dim, args.latent_dim).cuda()

  def forward(self, x):
    h = x
    for layer in self.layers:
      h = layer(h)
    mu = self.mu(h)
    logvar = self.var(h)
    if self.args.echo:
      mu = nn.Tanh()(mu)
      logvar = nn.Sigmoid()(logvar)
    return mu, logvar


class ConvDecoder(nn.Module):
  def __init__(self, D, args):
    super().__init__()
    self.layers = nn.ModuleList()
    self.args = args
    if args.decoder_architecture == 'conv':
      self.layers.append(nn.Linear(args.latent_dim, 256))
      #self.layers.append(nn.BatchNorm1d(num_features=256))
      self.layers.append(nn.ReLU())
      self.layers.append( Reshape([256,1,1]) )


      self.layers.append(nn.ConvTranspose2d(256, 64, 3, 1, 0, output_padding = 0))
      #self.layers.append(nn.BatchNorm2d(num_features=64))
      self.layers.append(nn.ReLU())

      self.layers.append(nn.ConvTranspose2d(64, 32, 3, 2, 0, output_padding = 0))
      #self.layers.append(nn.BatchNorm2d(num_features=32))
      self.layers.append(nn.ReLU())

      self.layers.append(nn.ConvTranspose2d(32, 32, 4, 2, 1, output_padding = 0))
      #self.layers.append(nn.BatchNorm2d(num_features=32))
      self.layers.append(nn.ReLU())

      self.layers.append(nn.ConvTranspose2d(32, 1, 4, 2, 1, output_padding = 0))
      self.layers.append( Flatten() )
      self.layers.append( nn.Sigmoid() )


      # {"type": "fc", "dim": 256, "activation": "relu", "batch_norm": true},
      # {"type": "reshape", "shape": [256, 1, 1]},
      # {"type": "deconv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 0, "output_padding": 0, "activation": "relu", "batch_norm": true},
      # {"type": "deconv", "filters": 32, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1, "activation": "relu", "batch_norm": true},
      # {"type": "deconv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "relu", "batch_norm": true},
      # {"type": "deconv", "filters": 3, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "sigmoid"}

    elif args.decoder_architecture is not None:
      raise NotImplementedError
    else:
      for i in range(args.num_deterministic_layers+1):
        if i < args.num_deterministic_layers:
          self.layers.append(nn.Linear(args.latent_dim if i == 0 else args.hidden_dim, \
                        args.hidden_dim))
                        
          self.layers.append(args.non_linearity)
        else:
          self.layers.append(nn.Linear(args.hidden_dim, D))
          self.layers.append(nn.Sigmoid())

  def forward(self, z):
    h = z 
    for layer in self.layers:
      h = layer(h)
    return h





# {
#   "decoder": [
#     {"type": "fc", "dim": 256, "activation": "relu", "batch_norm": true},
#     {"type": "reshape", "shape": [256, 1, 1]},
#     {"type": "deconv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 0, "output_padding": 0, "activation": "relu", "batch_norm": true},
#     {"type": "deconv", "filters": 32, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1, "activation": "relu", "batch_norm": true},
#     {"type": "deconv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "relu", "batch_norm": true},
#     {"type": "deconv", "filters": 3, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "sigmoid"}
#   ],
#   "encoder": [
#     {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
#     {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
#     {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 2, "padding": 0, "activation": "relu", "batch_norm": true},
#     {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 0, "activation": "relu", "batch_norm": true},
#     {"type": "flatten"},
#     {"type": "gaussian", "dim": 128}
#   ],
#   "transforms": AISConfig1,
#   "hidden_dim": 128
# }