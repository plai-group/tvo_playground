class AIS(nn.Module):

    '''
    Arguments
    ----------
		transforms : List

    '''
	def __init__(self, transforms, args):

		
		
		# should be flexible?
		self.partition = args.partition