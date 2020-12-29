
from torch import nn
import src.ais_flows.flows.hmc as hmc

'''Should these live in ais_flows folder somewhere? '''
'''
Can be used to specify common combinations of Flow transformations in form which is readable by ConfigParser of FlowComposition
e.g. nvp1_hmc2_mh = RealNVP (z) => RealNVP(v) => HMC => HMC => MH
'''


class HVAE_Block(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args

    self.layers = nn.ModuleList()

    self.layers.append( HMC(self.args) )
    # how to specify / feed in the MomentumRescaling?
    self.layers.append( MomentumRescaling(args, mom) )


