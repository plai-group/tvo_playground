   
'''https://github.com/vmasrani/alpha_ais/blob/master/src/hmc.py
'''

class InterpolatedEnergy(object):
	''' Parameters 
	    ------------
	    log_pi0 : function which takes (z, x) arguments
	    	+ log density of "proposal" or initial AIS density, unnormalized or normalized
	    log_p11 : function which takes (z, x) arguments
	    	+ log density "proposal" or initial AIS density, unnormalized or normalized
	'''
	
	def __init__(self, log_pi0, log_pi1, beta, q=1, kwargs0 = {}, kwargs1 = {}):
		self.log_pi0 = log_pi0
		self.log_pi1 = log_pi1
		self.beta = beta
		self.q = q 

		# just in case needed?  not sure
		self.kwargs0 = kwargs0	
		self.kwargs1 = kwargs1
		super().__init__()

	def log_prob(self, z, x = None):
		if self.q == 1:
			return (1-self.beta) * self.log_pi0(z, x)  + self.beta * self.log_pi1(z,x)
		else:
			return q_mixture(self.log_pi0, self.log_pi1)


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
