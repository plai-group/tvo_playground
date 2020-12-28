
from torch import nn
from src.handlers import ml_helpers as mlh
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

from src.models.base import ProbModelBaseClass

from src.utils.math_utils import log_mean_exp, sum_except_batch, singleton_repeat
from src.utils.losses import negative_bce
from src.utils.gaussian import log_normal_likelihood
#from dists import GaussianDist, GaussianEncoder, LayeredTransform
#from nflows import transforms, distributions, flows
from src.utils.schedules import get_partition
from types import SimpleNamespace
import src.handlers.trainers as trainers


def safe_encode(latents, size = None):
    if isinstance(latents, tuple):
        return latents[0], latents[1]
    else:
        size = size if size is None else int(latents.shape[-1]/2)
        return latents[:,:size], latents[:, size:]

class VAE(ProbModelBaseClass):
	'''
	Should be expanded to FlowVAE and base class (e.g. with init_opt, get_total_log_weight, maybe step_epoch)
	'''
	def __init__(self, args):
		# TO DO: Change signature of ProbModelBaseClass init
		super().__init__(args.input_dim, args)

		self.args = args
		
		self.internals = SimpleNamespace()		
		
		# TO DO: Replace with flexible architecture (see src/configs/architectures/vae_example.py)
		self.hidden_layer_dim = args.hidden_dim

		# Encoder (Replace with ConfigParser)
		self.enc1 = nn.Linear(args.input_dim, self.hidden_layer_dim)
		self.enc2 = nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)
		self.enc3 = nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)
		if self.args.echo:
			self.enc4_mu = nn.Linear(self.hidden_layer_dim, args.latent_dim)
			self.enc4_lv = nn.Linear(self.hidden_layer_dim, args.latent_dim)
		else:
			self.enc4 = nn.Linear(self.hidden_layer_dim, args.latent_dim * 2)
		# decoder



		self.encoder_flow, self.trf_list, self.encoder_dist = \
				create_flow(args, context_encoder = self.encode)


		# Decoder (Replace with ConfigParser)
		self.dec1 = nn.Linear(args.latent_dim, self.hidden_layer_dim)
		self.dec2 = nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)
		self.dec3 = nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)

	 
		self.recon_mu = nn.Linear(self.hidden_layer_dim, args.input_dim)
		self.observation_log_likelihood_fn = negative_bce
		self.recon_logvar = None # if we wanted to have a Gaussian observation model
	

		# Prior Architecture (Replace with ConfigParser)
		self.prior = None
		warnings.warn("Be more creative than having a Gaussian prior!   Implement flows or VampPrior")


		self.K = args.K
		
		self.partition = torch.tensor(get_partition(args)).cuda()    
		


	def init_optimizer(self):
		# Make optimizer
		if self.dual_objective:
			self.optimizer_phi_only = torch.optim.Adam(
				(params for name, params in self.named_parameters() if self.args.phi_tag in name), lr=self.args.lr)
			self.optimizer_theta_only = torch.optim.Adam(
				(params for name, params in self.named_parameters() if self.args.theta_tag in name), lr=self.args.lr)
		else:
			self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)


	def get_total_log_weight(self, S):
		with torch.no_grad():
			log_weight = []
			for obs in self.args.train_data_loader:
				self.set_internals(obs, S)
				elbo = self.elbo()
				log_weight.append(elbo)

			log_weight = torch.cat(log_weight)

		return log_weight


	def log_prior(self, z, sum_dim=None):
		if self.prior == 'vamp':
			pass
		elif self.prior=='flow': # more specific
			pass
		else:
			zeros = torch.zeros_like(z)
			return log_normal_likelihood(z, zeros, zeros, sum_dim=sum_dim)



	def step_epoch(self, data_loader, step=None):
		data_loader = tqdm(data_loader) if self.args.verbose else data_loader
		if self.dual_objective:
			return trainers.train_epoch_dual_objective(self, data_loader, self.optimizer, self.args, record=False)
			#return train_dual_objectives(data_loader)
		else:
			return trainers.train_epoch_single_objective(self, data_loader, self.optimizer, self.args, record=False)



	def encode(self, x, return_separate=True):
		''' TO DO: 

		Generalize : for each layer in architecture (i.e. encoder_layers), call its function?
		-----------
		'''

		if self.args.activation == 'relu':
			h1 = nnf.relu(self.enc1(x))
			h2 = nnf.relu(self.enc2(h1))
			h3 = nnf.relu(self.enc3(h2)) 
		else:
			h1 = torch.tanh(self.enc1(x))
			h2 = torch.tanh(self.enc2(h1))
			h3 = torch.tanh(self.enc3(h2))

		'''
			TO DO : Change to accommodate GaussianEncoder (or something which returns a density function stop_grad(mu, logvar) for reparam gradients (i.e. ignoring score function) )
		'''	
		if self.args.echo:
			mean = self.enc4_mu(h3)
			mean = torch.tanh(mean/16.0)

			logvar = self.enc4_lv(h3)
			logvar = torch.sigmoid(torch.exp(logvar))
		else:
			latent = self.enc4(h3)
			mean, logvar = latent[:, :self.args.z_size], latent[:, self.args.z_size:]

		if return_separate:
			return mean, logvar
		else:
			return latent

		# if self.args.stop_grad:
		# 	mean = mean.detach()
		# 	logvar = logvar.detach()


	def decode(self, z):
		''' TO DO: 
		
		Generalize : for each layer in architecture (i.e. decoder_layers), call its function?
		-----------
		'''
		if self.args.activation == 'relu':
			h1 = nnf.relu(self.dec1(z))
			h2 = nnf.relu(self.dec2(h1))
			h3 = nnf.relu(self.dec3(h2)) 
		else:
			h1 = torch.tanh(self.dec1(z))
			h2 = torch.tanh(self.dec2(h1))
			h3 = torch.tanh(self.dec3(h2))


		mean = self.recon_mu(h3) 
		return torch.sigmoid(mean), self.recon_logvar


	def _set_internals(self, x, S = None):
		S = S if S is not None else self.args.S 
		_ = self.elbo(x, S, set_internals=True)

    def set_internals(self, data, S=10):
        self.y = False  # VAEs are unsupervised
        self.x = data[0]
        self.inf_network = self.get_inf_network()
        if self.stop_grads:
            # self.inf_network does the sampling and self.inf_network_detached does the scoring
            self.inf_network_detached = self.get_inf_network(stop_grads=True)
        self.z = self.sample_latent(S)
        self.check_internals()

    def enable_stop_grads(self):
        self.stop_grads = True

    def disable_stop_grads(self):
        self.stop_grads = False

    def enable_reparam(self):
        self.reparam = True

    def disable_reparam(self):
        self.reparam = False


    def sample_latent(self, S):
    	# should rely on flows / AIS
    	raise NotImplementedError




	def elbo(self, x, S=1, set_internals=False):#return_all=False,  #return_all = None
		
		flattened_x = x.view(-1, self.args.input_dim)
		
		return_all_layers = 'tvo' in self.args.loss

		
		if self.args.num_betas==0:
			
			mu, logvar = self.encode(flattened_x)

			z, log_pz, log_prop = self._rsample(mu, logvar, S)
		else:

			z, log_prop = self.encoder_flow.sample_and_log_prob(S, \
									context = flattened_x, \
									return_all = return_all_layers)


		log_pz = self.log_prior(z, sum_dim = -1 if not return_all_layers else -2)
		

		# reshape & decode
		z = z.reshape(-1, self.args.z_size)
		x_mean, x_logvar = self.decode(z)#_flat)
		

		if return_all_layers:
			x_mean = x_mean.view(-1, self.args.input_dim, args.K ) if return_all_layers else x_mean
			x_logvar = x_logvar.view(-1, self.args.input_dim, args.K ) if x_logvar is not None else x_logvar

		x_target = singleton_repeat(flattened_x, S) 

		x_target = x_target.unsqueeze(-1) if return_all_layers else x_target
		#if len(x_target.shape) < len(x_mean.shape) else x_target
		
		log_lkd = self.observation_log_likelihood_fn( x_true = x_target, \
													 x_pred = x_mean,
													 x_logvar = x_logvar) #.view( S, -1 )

		elbo = log_lkd + log_pz - log_prop

		# adjust for p(x|z)
		if set_internals:
			self.internals.elbo = elbo
			self.internals.log_lkd = log_lkd
			self.internals.log_pz = log_pz
			self.internals.log_prop = log_prop

			if 'tvo' in self.args.loss: # and len(z.shape)>3
				# q(z_t|x) needed for TVO energy, whereas log_prop = r(z|x) flow probabilities 
				self.internals.log_qzx = self.encoder_dist.log_prob(z)
			else:
				self.internals.log_qzx = log_prop
				
		return elbo


	def forward(self, x, S=None, set_internals = False, itr = None):
		
		S = S if S is not None else self.args.S 
		# RSAMPLE only done in GaussianDist / GaussianEncoder atm
		
		elbo = self.elbo(x, S, set_internals= False, itr=itr) #set_internals)

		if True:
			return elbo



 #    def train_single_objective(self, data_loader):
 #        train_logpx = 0
 #        train_elbo = 0
 #        for idx, data in enumerate(data_loader):
 #            self.optimizer.zero_grad()

 #            loss, logpx, elbo = self.forward(data)

 #            loss.backward()
 #            self.optimizer.step()

 #            if self.args.record:
 #                self.record_stats()

 #            train_logpx += logpx.item()
 #            train_elbo += elbo.item()

 #        train_logpx = train_logpx / len(data_loader)
 #        train_elbo = train_elbo / len(data_loader)

 #        if self.args.record:
 #            self.save_record()

 #        return train_logpx, train_elbo

	# def train_dual_objectives(self, data_loader):
 #        train_logpx = 0
 #        train_elbo = 0
 #        for idx, data in enumerate(data_loader):
 #            self.optimizer_phi_only.zero_grad()
 #            self.optimizer_theta_only.zero_grad()

 #            if self.args.loss == 'tvo_reparam': # p optimized using tvo
 #                wake_theta_loss = self.get_tvo_loss(data)
 #            elif self.args.loss == 'iwae_dreg': # p optimized using IWAE (DReG update is only for q)
 #                wake_theta_loss = self.get_iwae_loss(data)
 #            elif self.args.loss in ['wake-wake', 'wake-sleep']:
 #                wake_theta_loss = self.get_wake_theta_loss(data)
 #            else:
 #                raise ValueError(f"{self.args.loss} is an invalid loss")

 #            wake_theta_loss.backward()
 #            self.optimizer_theta_only.step()

 #            if self.args.loss in ['tvo_reparam']:
 #                sleep_phi_loss = self.get_tvo_reparam_loss(data)
 #                sleep_phi_loss.backward()
 #            elif self.args.loss == 'iwae_dreg':
 #                sleep_phi_loss = self.get_iwae_dreg_loss(data)
 #                sleep_phi_loss.backward()
 #            elif self.args.loss == 'wake-sleep':
 #                sleep_phi_loss = self.get_sleep_phi_loss(data)
 #                sleep_phi_loss.backward()
 #            elif self.args.loss == 'wake-wake':
 #                wake_phi_loss = self.get_wake_phi_loss(data)
 #                wake_phi_loss.backward()
 #            else:
 #                raise ValueError(f"{self.args.loss} is an invalid loss")

 #            self.optimizer_phi_only.step()

 #            logpx, elbo = self.get_test_metrics(data, self.args.valid_S)

 #            if self.args.record:
 #                self.record_stats()

 #            train_logpx += logpx.item()
 #            train_elbo += elbo.item()

 #        if self.args.record:
 #            self.save_record()

 #        train_logpx = train_logpx / len(data_loader)
 #        train_elbo = train_elbo / len(data_loader)

 #        return train_logpx, train_elbo
