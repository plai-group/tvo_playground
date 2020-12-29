from torch import nn

class TupleDist(nn.Module):
	def __init__(self, dist_z, dist_v, args):
		self.dist_z = dist_z
		self.dist_v = dist_v
		self.args = args
		super().__init__()


	# let generative model set the value of x
	def log_prob(self, samples):

		z = samples[0]
		v = samples[1] if len(samples)>1 else None

		
		log_prob_z = self.dist_z(z)
		log_prob_x = self.dist_v(v) if v is not None else None
		
		return (log_prob_z, log_prob_v)


	def log_prob_z(self, z):
		return self.dist_z(z)

	def log_prob_v(self, v):
		return self.dist_v(v)

	def sample_z(self, S):
		return self.dist_z.sample(S)

	def sample_v(self, S):
		return self.dist_v.sample(S)

	def sample(self, S):
		return (self.dist_z.sample(S), self.dist_v.sample(S))