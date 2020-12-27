import numpy as np
from autograd import grad
import scipy.stats as st
from scipy import stats
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm
from scipy.special import logsumexp
from autograd.scipy import stats as autograd_stats


class Student1D:
    def __init__(
        self, q=None, mean=None, variance=None, precision=None, precisionxmean=None, df = None
    ):
        self.df = (3-q)/(q-1) if (q is not None and q!=1) else df
        if self.df is None:
            raise ValueError("Please specify q parameter (q!=1) or df (degrees of freedom)")
        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = 1.0 / variance
            self.precisionxmean = mean / variance
        else:
            if precisionxmean is not None:
                self.precision = precision
                self.precisionxmean = precisionxmean
                self.variance = 1.0 / precision
                self.mean = precisionxmean / precision
            else:
                self.mean = 0
                self.variance = 1
                self.precision = 1.0 / variance
                self.precisionxmean = mean / variance
        self.sigma = np.sqrt(self.variance)
        # RV with functions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
        #    e.g. logpdf(x, df, loc=0, scale=1),
        #         rvs(df, loc=0, scale=1, size=1, random_state=None) (sampling)
        self.student_t = stats.t
        self.student_t_autograd = autograd_stats.t
    def sample(self, no_samples):
        return self.student_t.rvs(self.df, size=no_samples, loc = self.mean, scale = self.sigma)
    def logprob(self, x):
        return self.student_t_autograd.logpdf(x, self.df, loc = self.mean, scale = self.sigma)


class StudentMV:
    def __init__(
        self, mean=None, variance=None, precision=None, precisionxmean=None
    ):
        raise NotImplementedError

    def sample(self, no_samples):
        raise NotImplementedError

    def logprob(self, x):
        raise NotImplementedError