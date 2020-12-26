


# Splines might be treated as functions which are utilized by a FlowLayer (i.e. can be used in RealNVP/coupling or MAF-type flows)

def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    '''
    Examples
    ----------
        https://github.com/bayesiains/nflows/blob/75048ff2ebd6b7ccad2fb8380630da08aa6ab86b/nflows/transforms/splines/rational_quadratic.py#L64
        https://github.com/noegroup/stochastic_normalizing_flows/blob/main/snf_code/snf_code/flowtools.py
    '''


# this might go in flow_layers.maf rather than here

# from flow_utils.spline import rational_quadratic_spline

# class MAF_RationalQuadraticSpline(FlowLayer):
    
#     '''
#     Examples
#     ----------
#         https://github.com/bayesiains/nflows/tree/master/nflows/transforms/splines
#     '''