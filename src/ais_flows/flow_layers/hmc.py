
from losses.energies import InterpolatedEnergy
# Change to inherit from Flow Class:  class HMC(Flow2D): or  class HMC(FlowLayer):

class HMC(nn.Module):

    '''
    Arguments
    ----------
    	energy_z : function or InterpolatedEnergy  (taking input (z, x = None) )
			"energy function" for which we follow gradients
			*** energy might imply -log πβ, but let's keep this as log πβ unless confusing
		
		Unnecessary to use energy_v (MH is separate operation)
	
		epsilon : float
			step size

		num_leapfrog : int
			number of leapfrog steps (should this be specified elsewhere)

    '''
	def __init__(self, energy_z, #energy_v, \
					   #epsilon, num_leapfrog, \
					   #adaptive_step_size_args = None, \
					   args):

		# SPECIFICATION OF THESE SHOULD CHANGE
		self.epsilon = args.hmc_epsilon # should be adaptive tensor for each dim of Z 
		self.num_leapfrog = args.num_leapfog

		self.energy_z = energy_z
		super().__init__()

	def forward(self, initial_samples, x, energy_z = None):

	    '''
	    Arguments
	    ----------
		initial_samples = (initial_z, initial_v) : tensors
			assume initial point of trajectory is given 
				(TO DO: incorporate sampling here if useful)
		
		energy_z : losses.energies.InterpolatedEnergy
			option to replace (if necessary?)

		Returns
		----------
		final_samples = (current_z, current_v) : tensors

		Δ log density ( = 0 for hamiltonian dynamics)

		'''
		current_z = initial_samples[0]
		current_v = intital_samples[1]

		grad_Uz, _ = self.energy_z.grad(current_z, x)

		v = current_v - 0.5 * grad_Uz * self.epsilon

		for i in range(1, self.num_leapfrog + 1):

		    z = z + v * self.epsilon

		    #grad_U_temp = get_grad_U_temp(z, batch, t, **kwargs)
		    grad_Uz, _ = self.energy_z.grad(current_z, x)

		    if i < self.num_leapfrog:
		        v = v - grad_Uz * self.epsilon

		v = v - 0.5 * grad_Uz * self.epsilon
		
		# momentum reversal as separate step?
		#v = -v

		# why this?
		# if not hparams.model_name == 'prior_vae':
		#     z.detach_()
		#     v.detach_()
		return (z, v), torch.sum(torch.zeros_like(z), dim=-1)



# Change to inherit from Flow Class:  class MomentumResampling(FlowLayer):

class MomentumResampling(nn.Module):
	def __init__(self, args):
		super().__init__()




class Langevin(nn.Module):

    '''
    Arguments
    ----------
    	target_z : function or InterpolatedEnergy  (taking input (z, x = None) )
			"energy function" for which we follow gradients, except energy implies -log πβ and may be confusing?
		
		target_v : function   (taking input (v, x = None) )
			log density for momentum variables
			(may also need to accomodate momentum re-sampling)
		
		initial_z, initial_v : tensors
			assume initial point of trajectory is given 
				(TO DO: incorporate sampling here if useful)

		epsilon : float
			step size

    '''
	def __init__(self, target_z, target_v, \
					   initial_z, initial_v, \
					   epsilon, adaptive_step_size_args = None, \
					   args):
		super().__init__()