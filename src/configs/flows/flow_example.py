

class FlowArch(object):
	def __init__(self, args):

		super().__init__()

	def	load_from_dict(self, dict):
		



from flow_layers import *
example_dict = [
		"flow_layers.MaskedAffineAutoregressiveTransform": {'layer_size': 100, 'num_bins':10, 'kwargs':{}},  
		"flow_layers.MaskedAffineAutoregressiveTransform": {'layer_size': 100, 'num_bins':10, 'kwargs':{}}, 
		"flow_layers.RandomPermutation": {}
	]