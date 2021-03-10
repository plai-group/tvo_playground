# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:51:04 2020

@author: Vu Nguyen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
from scipy.optimize import minimize
from src.gp.gptv import GPTV
from src.gp.gptv_perm import GPTV_Perm

from src.gp.gp import GaussianProcess
import matplotlib.pyplot as plt

import time
from sklearn.preprocessing import MinMaxScaler


counter = 0

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BayesOpt:

    def __init__(self, func, SearchSpace, GPtype="timevarying_perm",verbose=1):
        """
        Input parameters
        ----------

        SearchSpace:    SearchSpace on parameters defines the min and max for each parameter
        func:           a function to be optimized
        GPtype:         "timevarying_perm" #default value: time-varying permutation
                        "vanillaGP" #this is the vanilla GP, without timevarying permutation

        Returns
        -------
        dim:            dimension
        SearchSpace:         SearchSpace on original scale
        scaleSearchSpace:    SearchSpace on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.SearchSpace=SearchSpace
        self.dim = SearchSpace.shape[0]

        # for normalizing the input X [0,1]
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler

        # create a scaleSearchSpace 0-1
        self.scaleSearchSpace=np.array([np.zeros(self.dim), np.ones(self.dim)]).T

        # function to be optimised
        self.f = func

        # store X in original scale
        self.X_ori= None

        # store X in 0-1 scale
        self.X = None

        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None

        # y original scale
        self.Y_ori = None

        self.time_opt=0

        if "timevarying_perm" in GPtype:# timevarying, permutation
            self.gp=GPTV_Perm(self.scaleSearchSpace,noise_delta=1e-3,verbose=1)
        elif GPtype=="vanillaGP": # vanilla GP
            self.gp=GaussianProcess(self.scaleSearchSpace,noise_delta=1e-3,verbose=1)
        else:
            # time-varying, but not permutation invariant
            self.gp=GPTV(self.scaleSearchSpace,noise_delta=1e-3,verbose=1)



        # acquisition function
        self.acq_func = None
        self.logmarginal=0


    def init(self, n_init_points=3,seed=1):
        """
        Input parameters
        ----------
        gp_params:            Gaussian Process structure
        n_init_points:        # init points
        """

        np.random.seed(seed)

        init_X = np.random.uniform(self.SearchSpace[:, 0], self.SearchSpace[:, 1],size=(n_init_points, self.dim))

        self.X_original = np.asarray(init_X)

        # Evaluate target function at all initialization
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_ori = np.asarray(y_init)

        #self.Y_original_maxGP=np.asarray(y_init)
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_original)

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.SearchSpace[:,0]),self.max_min_gap)

        self.X = np.asarray(temp_init_point)


    def init_with_data(self, init_X,init_Y,isPermutation=False):
        """
        Input parameters
        ----------
        gp_params:            Gaussian Process structure
        x,y:        # init data observations (in original scale)
        """

        init_Y=(init_Y-np.mean(init_Y))/np.std(init_Y)

        # outlier removal
        # after standardise the data, remove outlier >3 and <-3
        # this is for robustness and only happen occasionally

        idx1=np.where( init_Y<=3)[0]
        init_Y=init_Y[idx1]
        init_X=init_X[idx1]

        idx=np.where( init_Y>=-3)[0]
        init_X=init_X[idx]
        init_Y=init_Y[idx]

        self.Y_ori = np.asarray(init_Y)
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)

        self.X_ori=np.asarray(init_X)
        self.X = self.Xscaler.transform(init_X)



    def gp_ucb(self,xTest):
        xTest=np.reshape(xTest,(-1,self.dim))
        mean, var,_,_ = self.gp.predict(xTest)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T

        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t =np.log(len(self.gp.Y))

        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return mean + np.sqrt(beta_t) * np.sqrt(var)


    def select_next_point(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        #ur = unique_rows(self.X)
        self.Y=np.reshape(self.Y,(-1,1))
        self.gp.fit(self.X, self.Y)

        # Set acquisition function
        start_opt=time.time()
        x_max = self.acq_max_scipy(ac=self.gp_ucb)

        x_max_ori=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim)))

        if self.f is None:
            return x_max,x_max_ori

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))

        # store X
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        # compute X in original scale

        self.X_ori=np.vstack((self.X_ori, x_max_ori))
        # evaluate Y using original X

        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_ori = np.append(self.Y_ori, self.f(x_max_ori))

        # update Y after change Y_original
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)

        return x_max,x_max_ori

    def acq_max_scipy(self,ac):
        """
        A function to find the maximum of the acquisition function using
        multi-start L-BFGS-B

        Input Parameters
        ----------
        ac: The acquisition function object that return its point-wise value.
        gp: A gaussian process fitted to the relevant data.
        y_max: The current maximum known value of the target function.
        SearchSpace: The variables SearchSpace to limit the search of the acq max.

        Returns
        -------
        x_max, The arg max of the acquisition function.
        """

        # Start with the lower bound as the argmax
        x_max = self.scaleSearchSpace[:, 0]
        max_acq = None

        #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}

        # this is the config of L-BFGS-B
        myopts ={'maxiter':50*self.dim,'maxfun':50*self.dim}

        # multi start: the number of repeatition is 3* dimension
        # higher dimensions will have more repetitions
        for i in range(3*self.dim):
            # Find the minimum of minus the acquisition function
            x_tries = np.random.uniform(self.scaleSearchSpace[:, 0], self.scaleSearchSpace[:, 1],size=(10*self.dim, self.dim))

            # evaluate the acquisition function
            y_tries=ac(x_tries)

            # pick the one with the best value and start L-BFGS-B from this point
            x_init_max=x_tries[np.argmax(y_tries)]

            res = minimize(lambda x: -ac(x.reshape(1, -1)),x_init_max.reshape(1, -1),
                   bounds=self.scaleSearchSpace,method="L-BFGS-B",options=myopts)#L-BFGS-B

            val=ac(res.x)

            # Store it if better than previous minimum(maximum).
            if max_acq is None or val >= max_acq:
                if 'x' not in res:
                    x_max = res
                else:
                    x_max = res.x
                max_acq = val

        return np.clip(x_max, self.scaleSearchSpace[:, 0], self.scaleSearchSpace[:, 1])


    def plot_acq_1d(self):
        # plot the acquisition function in 1 dimension

        x1_scale = np.linspace(self.scaleSearchSpace[0,0], self.scaleSearchSpace[0,1], 60)
        x1_scale=np.reshape(x1_scale,(-1,1))
        acq_value = self.gp_ucb(x1_scale)

        x1_ori=self.Xscaler.inverse_transform(x1_scale)

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(1, 1, 1)

        # Plot the surface.
        CS_acq=ax.plot(x1_ori,acq_value.reshape(x1_ori.shape))
        ax.scatter(self.X_ori[:,0],self.Y[:],marker='o',color='r',s=130,label='Obs')

        ax.set_ylabel('Acquisition Function',fontsize=18)
        ax.set_xlabel('Beta',fontsize=18)
