

# Change to inherit from Flow Class:  class HMC(Flow2D): or  class HMC(FlowLayer):

class HMC(nn.Module):

    '''
    Arguments
    ----------
    	energy_z : function or InterpolatedEnergy  (taking input (z, x = None) )
			"energy function" for which we follow gradients
			*** energy might imply -log πβ, but let's keep this as log πβ unless confusing
		
		Unnecessary
		# energy_v : function   (taking input (v, x = None) )
		# 	log density for momentum variables
		# 		(accomodate momentum re-sampling here or in new transform?)
		
		initial_z, initial_v : tensors
			assume initial point of trajectory is given 
				(TO DO: incorporate sampling here if useful)

		epsilon : float
			step size

		num_leapfrog : int
			number of leapfrog steps (should this be specified elsewhere)

    '''
	def __init__(self, energy_z, #energy_v, \
					   initial_z, initial_v, \
					   epsilon, num_leapfrog, \
					   adaptive_step_size_args = None, \
					   args):

		self.energy_z = energy_z
		super().__init__()

	def forward(self,):

		grad_U_temp = anneal_ops.grad_U(z, batch, t, **kwargs)



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