
from losses.energies import InterpolatedDensity
# Change to inherit from Flow Class:  class HMC(Flow2D): or  class HMC(FlowLayer):


class HMC(nn.Module):

    '''
    Arguments (DEF FEEL FREE TO CHANGE)
    ----------
    	neg_energy : InterpolatedDensity  (taking input (z, x = None) )
			# Commented out.   Just pass on .forward?


			"energy function" for which we follow gradients
			*** I'd like to call this an energy, but that might imply -log πβ and be confusing 
			*** I found -log confusing to keep track of
			*** HMC below also looks non-standard bc adding gradients now
		
	
		epsilon : float
			step size

		num_leapfrog : int
			number of leapfrog steps (should this be specified elsewhere)

    '''
	def __init__(self, #neg_energy, #energy_v, \
					   #epsilon, num_leapfrog, \
					   #adaptive_step_size_args = None, \
					   args):
		super().__init__()
		
		# way of specifying these should change...
		self.epsilon = args.hmc_epsilon # should be adaptive tensor for each dim of Z 
		self.num_leapfrog = args.num_leapfog

		self.neg_energy = neg_energy
		

	def forward(self, initial_samples, x, neg_energy = None):

	    '''
	    Arguments
	    ----------
		initial_samples = (initial_z, initial_v) : tensors
			assume initial point of trajectory is given 
				(TO DO: incorporate sampling here if useful)
		
		neg_energy : losses.energies.InterpolatedDensity
			option to replace (if necessary?)

		Returns
		----------
		final_samples = (current_z, current_v) : tensors

		Δ log density ( = 0 for hamiltonian dynamics)

		'''
		if neg_energy is not None:
			self.neg_energy = neg_energy

		current_z = initial_samples[0]
		current_v = intital_samples[1]

		# see InterpolatedDensity.grad
		grad_Uz, _ = self.neg_energy.grad(current_z, x)

		# USING LOG DENSITY INSTEAD OF ENERGY
		v = current_v + 0.5 * grad_Uz * self.epsilon
		#v = current_v - 0.5 * grad_Uz * self.epsilon

		for i in range(1, self.num_leapfrog + 1):

		    z = z - v * self.epsilon
		    #z = z + v * self.epsilon

		    #grad_U_temp = get_grad_U_temp(z, batch, t, **kwargs)
		    grad_Uz, final_energy,  = self.neg_energy.grad(current_z, x)

		    if i < self.num_leapfrog:
		    	v = v + grad_Uz * self.epsilon
		        #v = v - grad_Uz * self.epsilon

        v = v + 0.5 * grad_Uz * self.epsilon
		#v = v - 0.5 * grad_Uz * self.epsilon
		
		# momentum reversal as separate step?  necessary or not?
		v = -v


		zero_density = torch.sum(torch.zeros_like(z), dim=-1)

		return (z, v), (zero_density, zero_density) #, self.final_log_weight




class MH(nn.Module):
    '''
    Metropolis-Hastings accept-reject step
    --------------------------------------
    '''
	def __init__(self, neg_energy, log_density_v, args):
	    '''

	    Arguments
	    ----------

		might save computation by taking initial / target energies from HMC transformations

	    '''
		super().__init__()


class MomentumReverse(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def forward(self, initial_samples):
		# reverse momentum, return (z, v), Δ density
		zero_density = torch.sum(torch.zeros_like(initial_samples[0]), dim=-1)
		return (initial_samples[0], -1*initial_samples[-1]), (zero_density, zero_density)



# resample Momentum using 
class MomentumResampling(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def forward(self, initial_samples):
		raise NotImplementedError()

		# get access to the momentum density
		new_momentum = self.momentum_density.sample( initial_samples[-1].shape[0] )
		return (initial_samples[0], new_momentum)
		

class MomentumRescaling(nn.Module):
	def __init__(self, args, momentum_rescale=None, rescale_strategy = 'fixed_quadratic'):
		super().__init__()
		#currently set for scalars, but might encode STRATEGY such as quadratic tempering from HVAE
		self.momentum_rescale = momentum_rescale if momentum_rescale is not None else args.momentum_rescale

		self.rescale_strategy = rescale_strategy
		self.beta_0 = args.beta_0 

		self.args = args

	def forward(self, initial_samples, momentum_rescale=None, k = 0):
		if self.rescale_strategy == 'fixed_quadratic':
			# do calculations 
			momentum_rescale = unknown_function_of(k / self.args.K , self.beta_0)
			raise NotImplementedError


		momentum_rescale = momentum_rescale if momentum_rescale is not None else self.momentum_rescale
		# reverse momentum, return (z, v), Δ density
		zero_density = torch.sum(torch.zeros_like(initial_samples[0]), dim=-1)

		ones = torch.ones_like(initial_samples[-1])
		# 1/rescale log density adjustment in each dimension
		rescale_density = -torch.sum(torch.log(ones*momentum_rescale), dim = -1)

		return (initial_samples[0], momentum_rescale*initial_samples[-1]), (zero_density, rescale_density)



class Langevin(nn.Module):

    '''
    Arguments
    ----------
	One-Leapfrog HMC (copy from above)

    '''
	def __init__(self, target_z, target_v, \
					   initial_z, initial_v, \
					   epsilon, adaptive_step_size_args = None, \
					   args):
		super().__init__()