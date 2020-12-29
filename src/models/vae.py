
from torch import nn
from src.utils import ml_helpers as mlh
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import warnings


import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions import Independent
import numpy as np
from functools import partial

from src.models.base import ProbModelBaseClass

from src.utils.math_utils import log_mean_exp, sum_except_batch, singleton_repeat
from src.losses.losses import negative_bce
from src.dists.gaussian import log_normal_likelihood
from src.dists.gaussian import ConditionalGaussian # GaussianDist,
from src.dists.generator import GenerativeJoint
#from nflows import transforms, distributions, flows
from types import SimpleNamespace
#import src.utils.trainers as trainers

#from src.configs.architecture.vae_example import *
import src.configs.architectures




class FlowVAE(ProbModelBaseClass):
	'''
	Should be expanded to FlowVAE and base class (e.g. with init_opt, get_total_log_weight, maybe step_epoch)
	'''
	def __init__(self, args):
		# TO DO: Change signature of ProbModelBaseClass init
		super().__init__(args.input_dim, args)

		self.args = args
		
		self.tensor_dims =  (args.batch_size, args.S, args.K, args.latent_dim) \
									if args.K_per_chain > 1 else \
							(args.K_per_chain, args.batch_size, args.S, args.K, args.latent_dim)


		# TO DO: Replace with flexible architecture (see src/configs/architectures/vae_example.py)
		self.hidden_layer_dim = args.hidden_dim

		# objects with .forward(_) methods, loaded (most likely) from src.configs.architecture
		# specify config in main, see base.py for import_config_object
		self._encoder =  self.import_config_object(self.args.enc_architecture, module='src.configs.architectures')
		self._decoder =  self.import_config_object(self.args.dec_architecture, module='src.configs.architectures')


		# similarly for flows?  Create blocks of flows in configs



		# Generative Model Specification 
		# Decoding likelihood ( for 0/1 data, modify for pixel intensity or Gaussian observation)
		self.observation_log_likelihood_fn = negative_bce


		# Prior Architecture (Replace with ConfigParser)
		self.prior =  self.import_config_object(self.args.prior_dist, module='src.dists')
		self.initial_momentum =  self.import_config_object(self.args.momentum_dist, module='src.dists')
		warnings.warn("Be more creative than having a Gaussian prior!   Implement flows or VampPrior")
		

		# encoder_dist set in set_internals
		self.encoder_dist = None 
		self.decoder_dist = GenerativeJoint(self.prior, self.decode, self.observation_log_likelihood_fn, self.args) \


		'''see sample_latent'''
		self.sample_chain = None


		# used for doubly reparameterized gradients
		self.stop_grads = False

		# can remove once tested (or summing lkds over latent_dim can be incorporated into log_probs)
		self.reduce_op = lambda x: x #lambda x : torch.mean(x, dim = -1) #
	

	def get_total_log_weight(self, S):
		with torch.no_grad():
			log_weight = []
			for obs in self.args.train_data_loader:
				self.set_internals(obs, S)
				elbo = self.elbo()
				log_weight.append(elbo)

			log_weight = torch.cat(log_weight)

		return log_weight



	# def step_epoch(self, data_loader, step=None):
	# 	data_loader = tqdm(data_loader) if self.args.verbose else data_loader
	# 	if self.dual_objective:
	# 		return self.train_dual_objectives(data_loader)
	# 		#return trainers.train_epoch_dual_objective(self, data_loader, self.optimizer, self.args, record=False)
	# 		#return train_dual_objectives(data_loader)
	# 	else:
	# 		return self.train_single_objective(data_loader)
	# 		#return trainers.train_epoch_single_objective(self, data_loader, self.optimizer, self.args, record=False)



	def encode(self, x):
		return self._encoder(x)
		#mean, logvar = latent[:, :self.args.latent_dim], latent[:, self.args.latent_dim:]
		#return mean, logvar

	def decode(self, z):
		''' TO DO: 
		
		Generalize : for each layer in architecture (i.e. decoder_layers), call its function?
		-----------
		'''
		recon = self._decoder(z)
		return recon



	def set_internals(self, data, S= None):
		S = self.args.S if S is None else S

		self.y = False  # VAEs are unsupervised
		self.x = data[0]


		# self.inf_network = self.get_inf_network()
		# if self.stop_grads:
		#     # self.inf_network does the sampling and self.inf_network_detached does the scoring
		#     self.inf_network_detached = self.get_inf_network(stop_grads=True)
		self.z0 = self.sample_latent(S)


		z_shape = self.z.shape
		self.z = self.z.reshape(-1, z_shape[-1])


		# HMC sample_chains run decoder forward pass.  Collect in sample_chain.x_pred and reuse
		self.x_pred = self.sample_chain.x_pred if self.sample_chain is not None else self.decode(self.z)

		#self.x_pred = self.flow.x_pred if self.flow is not None else self.decode(z)

		self.x_pred = self.x_pred.reshape( (*z_shape[:-1], -1) )

		self.z = self.z.reshape(z_shape)

		self.check_internals()


	def sample_latent(self, S):
		# should rely on flows / AIS
		self.encoder_dist = ConditionalGaussian(self.encode, self.args).forward(self.x)

		if self.stop_grads:
			print("DOUBLE CHECK use of Stop_grad partial")
			import IPython
			IPython.embed()
			self.encoder_dist.stop_grad = True
			self.encoder_density = partial(self.encoder_dist.log_prob, stop_grad = True)
		else:
			self.encoder_density = self.encoder_dist.log_prob


		self.sample_chain = None
		''' e.g. AIS ( initial_density = self.encoder_dist, \
				 	 	 target_dist = self.decoder_dist, \
				 		 args, context = x	
					 )

	    '''


		z0 = self.encoder_dist.sample(S)

		if self.sample_chain is not None:
			self.v0 = self.momentum_dist.sample(S)

			# let this return a full tensor of [batch, S, K, dim]
			self.z = self.sample_chain.forward( (z0, self.v0), self.x )

			# sample_chain has already performed ingredients for TVO Objective
			self.snis_weights = self.sample_chain.snis_weights
			self.integrand = self.sample_chain.integrand
		else:
			# TO DO: expand z0 for original TVO static SNIS reweighting
			self.z = z0


		return self.z



	def enable_stop_grads(self):
		self.stop_grads = True

	def disable_stop_grads(self):
		self.stop_grads = False

	def enable_reparam(self):
		self.reparam = True

	def disable_reparam(self):
		self.reparam = False


	'''
		TO DO : Be smarter about these... HMC / AIS should already calculate TVO Integrand
	'''

	def log_joint(self):
		return self.log_likelihood() + self.log_prior()

	def log_likelihood(self):
		return self.reduce_op(self.observation_log_likelihood_fn(self.x, self.x_pred))

	def log_proposal(self):
		return self.reduce_op(self.encoder_dist.log_prob(self.z))

	def log_prior(self):
		return self.reduce_op(self.prior.log_prob(self.z))

		# 	zeros = torch.zeros_like(z)
		# 	return log_normal_likelihood(z, zeros, zeros, sum_dim=sum_dim)

	def elbo(self):
		self.check_internals()
		
		if self.sample_chain is not None: 
			# then read the elbo (i.e. integrand, log weights) off from here
			pass
		 
		return self.log_joint() - self.log_proposal()



