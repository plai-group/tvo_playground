# TO DO : import from flows configs.
from src.configs.flows import *



# class Architecture(object):
#   def __init__(self, decoder_list, encoder_list):

    
{
  "decoder": [
    {"type": "fc", "dim": 256, "activation": "relu", "batch_norm": true},
    {"type": "reshape", "shape": [256, 1, 1]},
    {"type": "deconv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 0, "output_padding": 0, "activation": "relu", "batch_norm": true},
    {"type": "deconv", "filters": 32, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1, "activation": "relu", "batch_norm": true},
    {"type": "deconv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "relu", "batch_norm": true},
    {"type": "deconv", "filters": 3, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 0, "activation": "sigmoid"}
  ],
  "encoder": [
    {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
    {"type": "conv", "filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu", "batch_norm": true},
    {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 2, "padding": 0, "activation": "relu", "batch_norm": true},
    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 0, "activation": "relu", "batch_norm": true},
    {"type": "flatten"},
    {"type": "gaussian", "dim": 128}
  ],
  "transforms": FlowExample,
  "hidden_dim": 128
}