# -*- coding: utf-8 -*-
"""
Created on April 2020

@author: Vu Nguyen
"""
import numpy as np
#import matplotlib.pyplot as plt
#import warnings
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import scipy
#from scipy.special import erf
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from src.gp.gptv import GPTV


class GPTV_Perm(GPTV): #Gaussian process time-varying permutation
    def __init__ (self,SearchSpace,noise_delta=1e-6,noise_drv=1e-4,verbose=0):
        self.noise_delta=noise_delta
        self.noise_drv=noise_drv
        self.mycov=self.cov_RBF_time_set
        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]

        self.hyper={}
        self.hyper['var']=1 # standardise the data
        self.hyper['lengthscale']=0.1 #to be optimised
        self.hyper['epsilon']=0.1 #to be optimised

        return None

    def fit(self,X,Y):
        """
        Fit a Gaussian Process model
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        """
        self.X_ori=X # this is the output in original scale
        #self.X= self.Xscaler.transform(X) #this is the normalised data [0-1] in each column
        self.X=X
        self.Y_ori=Y # this is the output in original scale
        self.Y=(Y-np.mean(Y))/np.std(Y) # this is the standardised output N(0,1)

        if len(self.Y)%5==0:
            self.hyper['epsilon'],self.hyper['lengthscale'],self.noise_delta=self.optimise()         # optimise GP hyperparameters
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x !")

        self.L=scipy.linalg.cholesky(self.KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)


    def cov_RBF(self,x1, x2,hyper):
        """
        Radial Basic function kernel (or SE kernel)
        """
        return super(GPTV, self).cov_RBF(x1, x2,hyper)

    def cov_RBF_time(self,x1,x2,hyper):
        """
        Radial Basic function kernel (or SE kernel)
        product with time-varying function
        SE(x1,x2) x (1-epsilon)^0.5*|t1-t2|
        see https://arxiv.org/pdf/1601.06650.pdf
        """
        return super(GPTV, self).cov_RBF_time(x1, x2,hyper)


    def cov_RBF_time_set(self,x1,x2,hyper):
        """
        Radial Basic function kernel (or SE kernel)
        product with time-varying function
        SE(x1,x2) x (1-epsilon)^0.5*|t1-t2|
        see https://arxiv.org/pdf/1601.06650.pdf
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']
        eps=hyper['epsilon']

        if x1.shape[1]!=x2.shape[1]: # check the dimension
            x1=np.reshape(x1,(-1,x2.shape[1]))

        x1=np.sort(x1,axis=1) # sorting
        x2=np.sort(x2,axis=1) # sorting
        Euc_dist=euclidean_distances(x1,x2)
        RBF=variance*np.exp(-np.square(Euc_dist)/lengthscale)

        if x1.shape[0]==1: # K(xnew, X)
            #time_vector1=np.asarray([x2.shape[0]+1]) # we consider predicting at T+1 timestep
            time_vector1=np.asarray([1])
        else: # K(X,X) # the timesteps for X is from 0,1,2....N
            time_vector1=np.linspace(0,x1.shape[0],x1.shape[0]+1)
            time_vector1=time_vector1/(x1.shape[0]+1) #normalise 0-1
            time_vector1=time_vector1[:-1]

        time_vector1=np.reshape(time_vector1,(x1.shape[0],1))

        time_vector2=np.linspace(0,x2.shape[0],x2.shape[0]+1)
        time_vector2=time_vector2/(x1.shape[0]+1) #normalise 0-1
        time_vector2=time_vector2[:-1]
        time_vector2=np.reshape(time_vector2,(x2.shape[0],1))

        dists = pairwise_distances(time_vector1,time_vector2, 'cityblock')

        timekernel=(1-eps)**(0.5*dists)
        output=RBF*timekernel

        return output


    def log_llk(self,X,y,hyper_values):

        #print(hyper_values)
        #print(hyper_values)
        hyper={}
        hyper['var']=1
        hyper['lengthscale']=hyper_values[1]
        hyper['epsilon']=hyper_values[0]
        noise_delta=hyper_values[2]

        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*noise_delta
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")

        try:
            L=scipy.linalg.cholesky(KK_x_x,lower=True)
            alpha=np.linalg.solve(KK_x_x,y)

        except: # singular
            return -np.inf
        try:
            first_term=-0.5*np.dot(self.Y.T,alpha)
            W_logdet=np.sum(np.log(np.diag(L)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)

        #print(hyper_values,logmarginal)
        return np.asscalar(logmarginal)

    def optimise(self):
        """
        Optimise the GP kernel hyperparameters
        Returns
        x_t
        """
        opts ={'maxiter':200,'maxfun':200,'disp': False}

        # epsilon, ls, var, noise var
        bounds=np.asarray([[0,0.9],[5e-1,5],[1e-4,5e-1]])
        #bounds=np.asarray([[0,0.0],[1e-2,1],[1e-7,1e-5]])


        init_theta = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(200, 3))
        logllk=[0]*init_theta.shape[0]
        for ii,val in enumerate(init_theta):
            logllk[ii]=self.log_llk(self.X,self.Y,hyper_values=val) #noise_delta=self.noise_delta

        x0=init_theta[np.argmax(logllk)]

        res = minimize(lambda x: -self.log_llk(self.X,self.Y,hyper_values=x),x0,
                                   bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B

        if self.verbose:
            print("estimated [epsilon lengthscale noisevar]",res.x)

        return res.x


    def predict(self,Xtest,isOriScale=False):
        """
        ----------
        Xtest: the testing points  [N*d]

        Returns
        -------
        pred mean, pred var, pred mean original scale, pred var original scale
        """

        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)

        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))

        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))

        KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)

        mean=np.dot(KK_xTest_x,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_x.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        mean_ori=    mean*np.std(self.Y_ori)+np.mean(self.Y_ori)
        std=np.reshape(np.diag(var),(-1,1))

        std_ori=std*np.std(self.Y_ori)#+np.mean(self.Y_ori)

        return mean,std,mean_ori,std_ori

    def estimate_Lipschitz(self):
        xtest=np.random.uniform(self.SearchSpace[:,0],self.SearchSpace[:,1],(5000,self.dim))

        mean,std,mean_ori,std_ori = self.predict(xtest,isOriScale=True)

        return np.max(mean)

    def plot_1d(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)

        mean,std,mean_ori,std_ori = self.predict(x1_ori)

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(1, 1, 1)

        # Plot the surface.
        CS_acq=ax.plot(x1_ori,mean_ori.reshape(x1_ori.shape))
        ax.scatter(self.X_ori[:,0],self.Y_ori[:],marker='o',color='r',s=130,label='Obs')

        temp_xaxis=np.concatenate([x1_ori, x1_ori[::-1]])
        temp_yaxis=np.concatenate([mean_ori - 1.9600 * std, (mean_ori + 1.9600 * std)[::-1]])
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')
        ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')


        ax.set_ylabel('Utility f(beta)',fontsize=18)
        ax.set_xlabel('Beta',fontsize=18)

    def plot_2d(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 60)
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

        mean,std,mean_ori,std_ori = self.predict(X_ori)

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(1, 1, 1)

        # Plot the surface.
        CS_acq=ax.contourf(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape),origin='lower')
        CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower')
        ax.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=130,label='Obs')

        try:
            ax.scatter(self.Xdrv_ori[:,0],self.Xdrv_ori[:,1],marker='s',color='y',s=130,label='Der')
        except:
            print()

        ax.set_xlabel('Epoch',fontsize=18)
        ax.set_ylabel('Beta',fontsize=18)
        fig.colorbar(CS_acq, ax=ax, shrink=0.9)

    def plot_1d_mean_var(self):
        X_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)

        mean,std,mean_ori,std_ori = self.predict(X_ori)

        fig = plt.figure(figsize=(14,4))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)

        # Plot the surface.
        CS_acq=ax_mean.plot(X_ori,mean_ori.reshape(X_ori.shape))
        ax_mean.scatter(self.X_ori[:,0],self.Y_ori[:],marker='o',color='r',s=100,label='Obs')


        ax_mean.set_xlabel('Log of Beta',fontsize=18)
        ax_mean.set_ylabel('f(beta)',fontsize=18)
        ax_mean.set_title("GP Mean",fontsize=20)


        # Plot the surface.
        CS_var=ax_var.plot(X_ori,mean_ori.reshape(X_ori.shape))
        ax_var.scatter(self.X_ori[:,0],self.Y_ori,marker='o',color='r',s=100,label='Obs')


        temp_xaxis=np.concatenate([X_ori, X_ori[::-1]])
        temp_yaxis=np.concatenate([mean_ori - 1.9600 * std, (mean_ori + 1.9600 * std)[::-1]])
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')
        ax_var.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')



        ax_var.set_xlabel('Log of Beta',fontsize=18)
        ax_var.set_ylabel('f(beta)',fontsize=18)
        ax_var.set_title("GP Var",fontsize=20)
        fig.savefig("1d_mean_var.pdf")


    def plot_2d_mean_var(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 60)
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]


        mean,std,mean_ori,std_ori = self.predict(X_ori)

        fig = plt.figure(figsize=(13,6))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)

        # Plot the surface.
        CS_acq=ax_mean.contourf(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape),origin='lower')
        CS2_acq = ax_mean.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower')
        ax_mean.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=100,label='Obs')

        try:
            ax_mean.scatter(self.Xdrv_ori[:,0],self.Xdrv_ori[:,1],marker='s',color='y',s=100,label='Der')
        except:
            print()

        ax_mean.set_xlabel('Epoch',fontsize=18)
        ax_mean.set_ylabel('Beta',fontsize=18)
        ax_mean.set_title("GP Mean",fontsize=20)
        fig.colorbar(CS_acq, ax=ax_mean, shrink=0.9)



        # Plot the surface.
        CS_var=ax_var.contourf(x1g_ori,x2g_ori,std.reshape(x1g_ori.shape),origin='lower')
        CS2_var = ax_var.contour(CS_var, levels=CS_var.levels[::2],colors='r',origin='lower')
        ax_var.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=100,label='Obs')

        try:
            ax_var.scatter(self.Xdrv_ori[:,0],self.Xdrv_ori[:,1],marker='s',color='y',s=130,label='Der')
        except:
            print()

        ax_var.set_xlabel('Epoch',fontsize=18)
        ax_var.set_ylabel('Beta',fontsize=18)
        ax_var.set_title("GP Var",fontsize=20)
        fig.colorbar(CS_var, ax=ax_var, shrink=0.9)


    def plot_3d(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 60)
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

        mean,std,mean_ori,std_ori = self.predict(X_ori)

        fig = plt.figure(figsize=(12,7))
        ax = plt.axes(projection="3d")

        # Plot the surface.
        #ax.scatter(self.X_ori[:,0],self.X_ori[:,1],self.Y_ori,marker='o',color='r',s=130,label='Data')
        ax.plot_wireframe(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape), color='green')

        ax.set_xlabel('Epoch',fontsize=18)
        ax.set_ylabel('Beta',fontsize=18)
        ax.set_zlabel('f(x)',fontsize=18)
