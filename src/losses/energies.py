   
'''https://github.com/vmasrani/alpha_ais/blob/master/src/hmc.py
'''

class InterpolatedEnergy(object):
	''' Parameters 
	    ------------
	    log_pi0 : function which takes (z, x) arguments
	    	+ log density of "proposal" or initial AIS density, unnormalized or normalized
	    log_p11 : function which takes (z, x) arguments
	    	+ log density "proposal" or initial AIS density, unnormalized or normalized

	    beta : float
	    	mixing parameter between [0,1]

	    q : float
	    	specifies q-path (where q=1 is geometric mixture path)

	    kwargs0, kwargs1 : dict
	    	might be passed to energy functions log_pi0, log_pi1 if ever necessary
	'''
	
	def __init__(self, log_pi0, log_pi1, beta, args, q=1, kwargs0 = {}, kwargs1 = {}):
		self.log_pi0 = log_pi0
		self.log_pi1 = log_pi1
		self.beta = beta
		self.q = q 
		self.args = args

		# just in case needed?  not sure
		self.kwargs0 = kwargs0	
		self.kwargs1 = kwargs1
		super().__init__()

	def log_prob(self, z, x = None):
		if self.q == 1:
			return (1-self.beta) * self.log_pi0(z, x)  + self.beta * self.log_pi1(z,x)
		else:
			raise NotImplementedError
			return q_mixture(z, x, self.log_pi0, self.log_pi1, self.beta, self.q)

	def grad(self, z, x = None, multi_sample = False):
		# used to store result of torch.autograd.grad
		grad_outputs = torch.ones((z.shape[0],))

		if multi_sample: # HMC with K samples per chain
			init = self.log_pi0(z,x)
			target = self.log_pi1(z,x)
			w = target - init
			interpolated = (1-self.beta) * init + self.beta * target
			grad = torch.autograd.grad(interpolated, z, grad_outputs = grad_outputs)[0]
			# requires SNIS reweighting of individual energies 
			raise NotImplementedError()
		else:
			log_prob = self.log_prob(z, x)
			grad = torch.autograd.grad(log_prob, z, grad_outputs = grad_outputs)[0]

        # clip by norm to avoid numerical instability
        grad = torch.clamp(
            grad, -z.shape[0] *
            self.args.latent_dim * 100,
            z.shape[0] * self.args.latent_dim 
            * 100)
        

        grad.requires_grad_()
        return grad, log_prob

    def grad_and_log_prob(self, z, x = None, multi_sample = False):
    	return self.grad(z, x, multi_sample)



def q_mixture( z, x, log_pi0, log_pi1, beta, q):
	''' ROB ATTEMPT using pytorch instead of np.logaddexp'''
    log_init = log_pi0(z,x)
    log_target = log_pi1(z,x)

    log_1t =  torch.log( (1-t)*torch.ones_like(log_init) )
    log_t =  torch.log( t*torch.ones_like(log_target) )

    energy = 1.0/q * torch.logsumexp(\
    			torch.cat([(log_1t + q*log_init ).unsqueeze(0), \
    						(log_t + q*log_target).unsqueeze(0)]), \
    					dim=0)
    #energy = 1.0/q * torch.logaddexp( log1t + q*log_pz, log_t + q*log_pxz )
    return energy 

# def q_mixture(log_pi0, log_pi1, beta, q):
#     assert 0 <= beta <= 1
#     if beta == 0: return log_pi0
#     pow1, pow2 = 1.0 - beta, beta

#     def neg_logp(x):
#         if alpha == 1:
#             log_prob = pow1*log_pi0.logprob(x) + pow2*log_pi1.logprob(x)
#         else:
#             log_prob = (2/(1-alpha))*(np.logaddexp(
#                 np.log(pow1) + ((1-alpha)/2)* log_pi0.logprob(x),
#                 np.log(pow2) + ((1-alpha)/2)* log_pi1.logprob(x)))
#         return -log_prob

#     return HMCDist(neg_logp) 