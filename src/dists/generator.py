from torch import nn

class GenerativeJoint(nn.Module):
	def __init__(self, prior, decoder, likelihood, args):
		self.prior = prior
		self.decoder = decoder
		self.lkd = likelihood 
		self.args = args
		super().__init__()

	def log_prob(self, z, x=None, stop_grad_x = False):
		if x is None:
			x = self.decoder(z)

		log_pz = self.prior.log_prob(z)

		# NOTE: we will take gradients with respect to LKD function here
			 # check if this should be 'self.lkd.log_prob'
		log_pxgz = self.lkd(x).detach() if stop_grad_x else self.lkd(x)
			
		return torch.sum(log_pz, dim=-1) + torch.sum(log_pxgz, dim = -1)


	# TO DO : is returning z,x expected behavior?   don't do this elsewhere
	def sample(self, S):
		z = self.prior.sample(S)
		x = self.decoder(z)
		return z, x

	def sample_and_log_prob(self, S):
		z, x = self.sample(S)
		return (z,x), self.log_prob(z,x)