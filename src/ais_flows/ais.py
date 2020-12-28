from losses.energies import InterpolatedDensity

class Chain():
	def __init__(self, transforms, initial_density, target_density, args):

		self.transforms = transforms

		self.initial_density = initial_density
		self.target_density = target_density

		# should be flexible
		self.partition = args.partition

	def forward(self):
		raise NotImplementedError

	def get_proposal_weights(self):
		raise NotImplementedError

	def get_mh_weights(self):
		raise NotImplementedError

	def calc_snis_weights(self):
		if use_mh_estimation:
			self.get_mh_weights()
		else:
			self.get_proposal_weights()


'''
Subclasses of Chains might be used to directly specify compositions of transformations (i.e. architectures)
e.g.
AIS = all HMC
OneRNVP_TwoHMC
AllIAF (i.e. only IAF at each layer with flow-specific params)

'''


class AIS(nn.Module):

    '''
    Arguments
    ----------
		transforms : List
		
		initial_density : TupleDist with density for both z, v

		target_density :  TupleDist with density for both z, v

    '''
	def __init__(self, transforms, initial_density, target_density, args):
		super().__init__(transforms, initial_density, target_density, args)

		self.transforms = []


		for t in range(self.args.K):
			beta_t = args.partition[t]
			intermediate_density = InterpolatedDensity( initial_dist, target_dist, beta_t, args)


	def forward(self, initial_samples, x):
		# current_z = initial_samples[0]
		# current_v = initial_samples[-1] if len(initial_samples)>1 else None
		
		all_samples = [initial_samples]
		samples = initial_samples
		densities = self.initial_density()

		for transform in self.transforms:
			samples, density_change = transform(samples, x)
			all_samples.append(samples)