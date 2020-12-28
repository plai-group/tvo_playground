


'''
Can be used to specify common combinations of Flow transformations in form which is readable by ConfigParser of FlowComposition
e.g. nvp1_hmc2_mh = RealNVP (z) => RealNVP(v) => HMC => HMC => MH

These can then be combined for AIS configs ({ beta_1 : nvp1_hmc2_mh, beta_2: nvp1_hmc2_mh, beta_3:  nvp1_hmc2_mh })

'''


from flow_layers import *
example_dict = [
		"flow_layers.MaskedAffineAutoregressiveTransform": {'layer_size': 100, 'num_bins':10, 'kwargs':{}},  
		"flow_layers.MaskedAffineAutoregressiveTransform": {'layer_size': 100, 'num_bins':10, 'kwargs':{}}, 
		"flow_layers.RandomPermutation": {}
	]


class FlowArch(object):
	def __init__(self, args):

		super().__init__()

	def	load_from_dict(self, dict):
		


