from losses.energies import InterpolatedDensity
from ais_flows.flows import HMC

class Chain():
	def __init__(self, initial_density, target_density, args, transforms = None):

		self.initial_density = initial_density
		self.target_density = target_density

		self.transforms = transforms
		# should be flexible
		self.partition = args.partition

	def forward(self):
		raise NotImplementedError

	def get_proposal_weights(self):
		# e.g. using Sec 3.2-3.3 (SNIS with flow proposal)
		raise NotImplementedError

	def get_mh_weights(self):
		# e.g. using Sec 3.4 (AIS chain weights)
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
			( this is probably silly, split this up... )
		
		target_density :  TupleDist with density for both z, v


    '''
	def __init__(self, initial_density, target_density, \
					args, transforms = None, context= None):
	# def __init__(self, initial_density_z, initial_density_v, 
	# 					target_density_z, target_density_v, \
	# 					args, transforms = None, context= None):
		super().__init__(transforms, initial_density, target_density, args)

		self.transforms = transforms
		self.args = args 
		
		#z_size = (args.batch_size, args.S, args.K, args.latent_dim) \
		#			if args.K_per_chain > 1 else \
		#			(args.K_per_chain, args.batch_size, args.S, args.K, args.latent_dim)

		# we will append and concatenate along dim -2 for args.K
		self.x = context
		self.context = context

		self.z = [] 		# append, then torch.cat([z0.unsqueeze(-2), z1.unsqueeze(-2), z2.unsqueeze(-2)], dim =-2)
		self.v = [] 			# yields [batch, S, K, dim]
		self.x_pred = []	# similarly collect forward decoder passes of HMC for reuse (needed?)

		# density adjustments along the chain
		self.proposal_densities = []

		# self.pi_betas store integrand = log p/q from forward pass of HMC transformations!
		self.pi_betas = nn.ModuleList()

		for t in range(args.K):
			beta_t = args.partition[t]
			self.pi_betas.append( InterpolatedDensity( initial_density.log_prob_z, target_dist.log_prob_z, beta_t, args) )


	def forward(self, initial_samples, x = None):
		'''
			Interpolated samples 
		'''

		# current_z = initial_samples[0]
		# current_v = initial_samples[-1] if len(initial_samples)>1 else None
		self.x = self.x if x is None else x
		
		samples = initial_samples
		densities = self.initial_density()

		

		# instead of specifying list of transform (blocks directly )
		# for transform in self.transforms:
		# 	samples, density_change = transform(samples, self.x)
		# 	all_samples.append(samples)


		# for HMC, avoid re-doing forward passes for TVO integrand since the InterpolatedDensity already evaluates this
		self.final_log_weight = self.pi_betas[i]._log_weight

		# can be used to avoid recalculating energy in Metropolis-Hastings
		self.final_neg_energy = self.pi_betas[i]._log_prob


	def calc_snis(self):
		raise NotImplementedError










