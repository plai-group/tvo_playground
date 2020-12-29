
import torch
from torch import nn

'''
UNUSED, UNTESTED
'''


class Composition(nn.Module): #transforms.Transform):
    '''Composes several transforms into one, in the order they are given.

        Inputs
        ------
        transforms: an iterable of `FlowLayer` objects

        Function
        --------
        log_det_jac : outputs change in density for (forward or reverse) transformation
        
        forward : almost equivalent to sample_and_log_prob (apart from initial density)
        inverse :

        sample_and_log_prob ( initial_z , initial_v, initial_density ):
            e.g. let the AIS object sample the initial z, momentum
            each FlowComposition transforms and 


    '''

    def __init__(self, transforms):
        """Constructor.
        Args:
            
        """
        super(Composition, self).__init__()
        self._transforms = nn.ModuleList(transforms)


    def log_det_jac(self, tbd):
        raise NotImplementedError

    # redefine arguments to be more z, x, etc. focused
    def forward(self, inputs, context=None, return_all = True):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context, return_all = return_all)

    def inverse(self, inputs, context=None, return_all = True):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, return_all = return_all)


    def sample_and_log_prob(self, initial_z, initial_v, initial_density):
        raise NotImplementedError
        # samples, change_in_density = self.forward
        # log_prob = initial_density + change_in_density

    @staticmethod
    def _cascade(inputs, funcs, context, return_all = True):

        '''UPDATE to better naming convention for inputs(z, possibly v), context (x) , functions (flow layers)'''

        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        if return_all:
            returns = []
            logabsall = []
            # elemnts of returns (batch x z_size x 1), logabsall  ( batch x 1 ) have extra dim for concat over layers
            returns.append(outputs.unsqueeze(-1))
            logabsall.append(torch.zeros_like(utils.sum_except_batch(inputs).unsqueeze(-1)))

        for i in range(len(funcs)):
            func = funcs[i]

            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet

        if return_all and returns:
            returns = torch.cat(returns, dim = -1)
            logabsall = torch.cumsum( torch.cat(logabsall, dim =-1) , -1)
            return returns, logabsall
        else:
            return outputs, total_logabsdet