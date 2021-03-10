# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:35:25 2020

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.special import erf

class GaussianProcess(object):
    def __init__ (self,SearchSpace,noise_delta=1e-8,noise_drv=1e-4,verbose=0):
        self.noise_delta=noise_delta
        self.noise_drv=noise_drv
        self.mycov=self.cov_RBF
        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]
        
        self.hyper={}
        self.hyper['var']=1 # standardise the data
        self.hyper['lengthscale']=0.1 #to be optimised
        
        #self.minY=0 # this is the value of f(beta=0)

        return None

        
    def fit(self,X,Y):
        """
        Fit a Gaussian Process model
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        """       
        #Y=Y-self.minY
        self.X_ori=X # this is the output in original scale
        self.X= self.Xscaler.transform(X) #this is the normalised data [0-1] in each column
        self.Y_ori=Y # this is the output in original scale
        self.Y=(Y-np.mean(Y))/np.std(Y) # this is the standardised output N(0,1)
        
        if len(self.Y)%5==0:
            self.hyper['lengthscale']=self.optimise()         # optimise GP hyperparameters
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x !")
      
        self.L=scipy.linalg.cholesky(self.KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        
    def fit_drv(self,X,Y,Xdrv,Ydrv,drv_index=0):
        """
        Fit a Gaussian Process model using derivative
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        Xdrv: input 2d array of derivative obs [M*d]
        Ydrv: output 2d array of derivative obs [M*1]
        drv_index: the derivative is applicalbe at drv_index (in this case drv_index=0)
        """    
        #Y=Y-self.minY

        self.drv_index=drv_index
        self.Xdrv_ori=np.reshape(Xdrv,(-1,X.shape[1]))
        self.Xdrv=self.Xscaler.transform(Xdrv) #normalised the derivative input [0-1]
        self.X_ori=X
        self.X= self.Xscaler.transform(X) # normalised the input [0-1]
        self.Y_ori=Y
        self.Ydrv_ori=np.copy(Ydrv )
        self.Ydrv=Ydrv/np.std(Y) # standardize derivative given the data
        #print("Ydrv_ori",Ydrv)
        #print("Ydrv",self.Ydrv)
        #self.Ydrv=np.copy(Ydrv )
        #print("stdY",np.std(Y))
        self.Y=(Y-np.mean(Y))/np.std(Y)
        
        if self.verbose:
            print("Y",self.Y)
        self.Y_combined=np.vstack((self.Y,self.Ydrv))
        self.hyper['lengthscale']=self.optimise_drv()         # optimise GP hyperparameters

        KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")
            
        KK_x_v=self.cov_RBF_drv(self.Xdrv,self.X,self.hyper,drv_index).T

        flag=True
        #self.noise_drv=1
        while(flag==True and self.noise_drv<10):
            try:
                KK_v_v=self.cov_RBF_drv_drv_itself(self.Xdrv, self.hyper,drv_index)\
                            +np.eye(self.Xdrv.shape[0])*self.noise_drv
                top_matrix=np.hstack((KK_x_x,KK_x_v))
                bottom_matrix=np.hstack((KK_x_v.T, KK_v_v))
                self.KK_combined_drv=np.vstack((top_matrix,bottom_matrix))
        
                self.Ldrv=scipy.linalg.cholesky(self.KK_combined_drv,lower=True)
                flag=False
            except:
                self.noise_drv=self.noise_drv*2 #double the noise in the covariance is not invertable
                
        if self.verbose:
            print(self.noise_drv)
                
        temp=np.linalg.solve(self.Ldrv,self.Y_combined)
        self.alphadrv=np.linalg.solve(self.Ldrv.T,temp)
        
        self.L=scipy.linalg.cholesky(KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        

    def cov_RBF(self,x1, x2,hyper):        
        """
        Radial Basic function kernel (or SE kernel)
        """
        
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)

        return variance*np.exp(-np.square(Euc_dist)/lengthscale)
  
    def cov_lin_RBF(self,x1, x2,hyper):
        return self.cov_linear(x1, x2)+self.cov_RBF(x1, x2,hyper)

    def log_llk(self,X,y,lengthscale,noise_delta=1e-5):
        
        hyper={}
        hyper['var']=1
        hyper['lengthscale']=lengthscale

        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*noise_delta     
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")   

        try:
            L=scipy.linalg.cholesky(KK_x_x,lower=True)
            alpha=np.linalg.solve(KK_x_x,y)

        except: # singular
            #print("singular",hyper['lengthscale'],noise_delta)
            return -np.inf
        try:
            #print("okay",hyper['lengthscale'],noise_delta)

            first_term=-0.5*np.dot(self.Y.T,alpha)
            W_logdet=np.sum(np.log(np.diag(L)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)
        return np.asscalar(logmarginal)
    
    
    def optimise(self):
        """
        Optimise the GP kernel hyperparameters
        Returns
        x_t
        """
        
        opts ={'maxiter':200,'maxfun':200,'disp': False}

        #x0=[0.01,0.02]
        bounds=np.asarray([[1e-2,1]])
        
        init_theta = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20, 1))
        logllk=[0]*init_theta.shape[0]
        for ii,val in enumerate(init_theta):           
            logllk[ii]=self.log_llk(self.X,self.Y,lengthscale=val,noise_delta=self.noise_delta)
            
        x0=init_theta[np.argmax(logllk)]

        res = minimize(lambda x: -self.log_llk(self.X,self.Y,lengthscale=x,noise_delta=self.noise_delta),x0,
                                   bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B
        
        if self.verbose:
            print(res.x)
            
        self.hyper['lengthscale']=res.x
        return res.x
    
  
    
    def predict_gradient(self,Xtest):
        """
        Predicting the gradient value at Xtest
        """
        Xtest=self.Xscaler.transform(Xtest)
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.cov_RBF_drv_drv_itself(Xtest,self.hyper,drv_index=0)+np.eye(Xtest.shape[0])*self.noise_drv
        KK_xTest_x=self.cov_RBF_drv(Xtest,self.X,self.hyper)

        mean_grad=np.dot(KK_xTest_x,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_x.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        mean_grad_ori= mean_grad*np.std(self.Y_ori)+np.mean(self.Y_ori)
        std_grad=np.reshape(np.diag(var),(-1,1))
        
        std_grad_ori=std_grad*np.std(self.Y_ori)#+np.mean(self.Y_ori)
        
        return mean_grad,std_grad,mean_grad_ori,std_grad_ori
    
   
    
    def predict(self,Xtest):
        """
        ----------
        Xtest: the testing points  [N*d]

        Returns
        -------
        pred mean, pred var, pred mean original scale, pred var original scale
        """    
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
    
    def predict_with_drv(self,Xtest):
        """
        ----------
        Xtest: the testing points [N*d]

        Returns
        -------
        pred mean, pred var, pred mean original scale, pred var original scale
        """    
        Xtest=self.Xscaler.transform(Xtest)
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)
        
        # compute KK_xTest_xdrv
        KK_xTest_xdrv=self.cov_RBF_drv(self.Xdrv,Xtest,self.hyper,self.drv_index).T
        
        KK_xTest_x_xdrv=np.hstack((KK_xTest_x,KK_xTest_xdrv))
        
        # using Cholesky update
        mean=np.dot(KK_xTest_x_xdrv,self.alphadrv)
        v=np.linalg.solve(self.Ldrv,KK_xTest_x_xdrv.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        mean_ori=    mean*np.std(self.Y_ori)+np.mean(self.Y_ori)
        std=np.reshape(np.diag(var),(-1,1))
        
        std_ori=std*np.std(self.Y_ori)#+np.mean(self.Y_ori)
        
        return mean,std,mean_ori,std_ori
    

 
    def plot_1d_drv(self,myfunction):
        # plot 1d function using derivative observations
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        Y_ori_at_drv=myfunction(self.Xdrv)
        pred_mean_drv,pred_std_drv,pred_mean_ori_drv,pred_std_ori_drv=self.predict_with_drv(Xtest)

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        CS_acq=ax.plot(Xtest,pred_mean_ori_drv.reshape(Xtest.shape),':k',linewidth=3,label='GP mean drv')
        temp_xaxis=np.concatenate([Xtest, Xtest[::-1]])
        temp_yaxis=np.concatenate([pred_mean_ori_drv - 1.9600 * pred_std_ori_drv, 
                                   (pred_mean_ori_drv + 1.9600 * pred_std_ori_drv)[::-1]])
        ax.scatter(self.X_ori,self.Y_ori,marker='s',s=100,color='g',label='True Obs')  
        ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')

        y_org=myfunction(Xtest)
        
        CS_acq=ax.plot(Xtest,y_org.reshape(Xtest.shape),'r',label="True Function")
        
        def line_from_gradient(gradient):        # helper function to plot the gradient vector
            myin=np.linspace(0,0.08,30)
            return gradient*myin

        for ii in range(len(self.Ydrv_ori)):
            mylbs=self.Xdrv[ii]-0.05
            myubs=self.Xdrv[ii]+0.03
            myin=np.linspace(mylbs,myubs,30)
            myout=line_from_gradient(self.Ydrv_ori[ii])
        
            if ii==0:
                plt.plot(myin, myout+Y_ori_at_drv[ii]+0.5, '-b', label='Gradient')
            else:
                plt.plot(myin, myout+Y_ori_at_drv[ii]+0.5, '-b')
        
        ax.legend(prop={'size': 14})
        
        ax.set_ylabel("Integrand",fontsize=16)
        ax.set_xlabel("Beta",fontsize=16)
        
        ax.set_title("MNIST TVO VAE",fontsize=20)  
        strPath="plot/MNIST_TVO_VAE_BQD_{:d}_points.pdf".format(len(self.Y))
        fig.savefig(strPath)
        
    def plot_1d(self,myfunction):
        # plot 1d function using derivative observations
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        pred_mean_drv,pred_std_drv,pred_mean_ori_drv,pred_std_ori_drv=self.predict(Xtest)

        fig = plt.figure(figsize=(8,5))
        
        ax = fig.add_subplot(1,1,1)
        
        #my_cmap = plt.get_cmap('Blues')
        CS_acq=ax.plot(Xtest,pred_mean_ori_drv.reshape(Xtest.shape),':k',linewidth=3,label='GP mean')
        
        temp_xaxis=np.concatenate([Xtest, Xtest[::-1]])
        temp_yaxis=np.concatenate([pred_mean_ori_drv - 1.9600 * pred_std_ori_drv, (pred_mean_ori_drv + 1.9600 * pred_std_ori_drv)[::-1]])
        ax.scatter(self.X_ori,self.Y_ori,marker='s',s=100,color='g',label='True Obs')  
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')

        y_org=myfunction(Xtest)
        
        CS_acq=ax.plot(Xtest,y_org.reshape(Xtest.shape),'r',label="True Function")
        
        
        ax.legend(prop={'size': 14})
        
        ax.set_ylabel("f(x)",fontsize=16)

        #ax.set_ylabel("Integrand",fontsize=16)
        ax.set_xlabel("Beta",fontsize=16)
        
        ax.set_title("Original Function f",fontsize=20)  
        #strPath="plot/MNIST_TVO_VAE_BQD_{:d}_points.pdf".format(len(self.Y))
        fig.savefig("plot/original_func.pdf")
  

    def plot_gradient_1d(self,myfunction):
        # plot 1d function using derivative observations
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        pred_grad,pred_std_of_grad,pred_grad_ori,pred_std_of_grad_ori=self.predict_gradient(Xtest)

        fig = plt.figure(figsize=(8,5))
        
        ax = fig.add_subplot(1,1,1)
        
        #my_cmap = plt.get_cmap('Blues')
        CS_acq=ax.plot(Xtest,pred_grad.reshape(Xtest.shape),':k',linewidth=3,label='Derivative of GPmean')
        
        temp_xaxis=np.concatenate([Xtest, Xtest[::-1]])
        temp_yaxis=np.concatenate([pred_grad - 1.9600 * pred_std_of_grad, (pred_grad + 1.9600 * pred_std_of_grad)[::-1]])
        ax.scatter(self.Xdrv,self.Ydrv,marker='s',s=100,color='g',label='Derivative Obs')  
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        #ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')

        #y_org=myfunction(Xtest)
        
        #CS_acq=ax.plot(Xtest,y_org.reshape(Xtest.shape),'r',label="True Function")
        
        ax.legend(prop={'size': 14})
        
        ax.set_ylabel("$\delta f(x)$",fontsize=16)
        ax.set_xlabel("Beta",fontsize=16)
        
        ax.set_title("Derivative Function f'",fontsize=20)  
        #strPath="plot/MNIST_TVO_VAE_BQD_{:d}_points.pdf".format(len(self.Y))
        fig.savefig("plot/der.pdf")