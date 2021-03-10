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

class GaussianProcessDerivative(object):
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

    #def set_min_Y(self,minY):
        #self.minY=minY
        
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
    
    def cov_RBF_drv(self,x1, x2,hyper, drv_index=0): # the first argument is drv       
        """
        x1: xdrv
        x2: x
        Radial Basic function kernel (or SE kernel)
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']
            
        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
            
        Euc_dist=euclidean_distances(x1,x2)
        k_x_xdrv=variance*np.exp(-np.square(Euc_dist)/lengthscale)
        
        temp1=np.atleast_2d(x1[:,drv_index]).T
        temp2=np.atleast_2d(x2[:,drv_index]).T
        temp=np.squeeze(temp1[:,None] - temp2)
        dist_x_xdrv_d=-temp*1.0/lengthscale
        dist_x_xdrv_d=np.reshape(dist_x_xdrv_d,(x1.shape[0],x2.shape[0]))
        return k_x_xdrv*dist_x_xdrv_d
    
    def cov_RBF_drv_drv_itself(self,xx,hyper, drv_index=0):        
        """
        second derivative of xx to itself at drv_index
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']
            
        temp1=np.atleast_2d(xx[:,drv_index]).T
        temp2=np.atleast_2d(xx[:,drv_index]).T
        
        Euc_dist_xdrv_xdrv=euclidean_distances(temp1,temp2)
        K_xdrv_xdrv=variance*np.exp(-np.square(Euc_dist_xdrv_xdrv)/lengthscale)
        
        temp=np.squeeze(temp1[:,None] - temp2)
        
        #dist_xdrv_xdrv_d=(temp*temp.T)/lengthscale
        dist_xdrv_xdrv_d=np.multiply(temp,temp)/lengthscale


        #return np.multiply(K_xdrv_xdrv,(np.eye(xx.shape[0])-dist_xdrv_xdrv_d))/lengthscale
        return np.multiply(K_xdrv_xdrv,(1-dist_xdrv_xdrv_d))/lengthscale

    def cov_RBF_drv_drv(self,xx1,xx2,hyper, drv_index=0):        
        """
        second derivative of xx1 vs xx2 at drv_index
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']
        
        temp1=np.atleast_2d(xx1[:,drv_index]).T
        temp2=np.atleast_2d(xx2[:,drv_index]).T
        
        Euc_dist_xdrv_xdrv=euclidean_distances(temp1,temp2)
        K_xdrv_xdrv=variance*np.exp(-np.square(Euc_dist_xdrv_xdrv)/lengthscale)
        
        temp12=np.squeeze(temp1[:,None] - temp2)
        #temp21=np.squeeze(temp2[:,None] - temp1)

        dist_xdrv_xdrv_d=np.multiply(temp12,temp12)/lengthscale
        #dist_xdrv_xdrv_d=(temp12*temp21.T)/lengthscale

        return np.multiply(K_xdrv_xdrv,(1-dist_xdrv_xdrv_d))/lengthscale
    
    def cov_RBF_integral(self,xx1,xx2,hyper,integral_idx=0):
        """
        estimate the \int k(x1,x2) dx1 at dimension integral_idx
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        temp1=np.atleast_2d(xx1[:,integral_idx]).T
        temp2=np.atleast_2d(xx2[:,integral_idx]).T
        
        Euc_dist_xint_x=euclidean_distances(temp1,temp2)
        #K_xint_x=variance*np.exp(-np.square(Euc_dist_xint_x)/lengthscale)
        
        return 0.5*np.sqrt(3.14*lengthscale)*variance*erf(Euc_dist_xint_x/np.sqrt(lengthscale))
    
    def cov_RBF_integral_integral(self,xx1,xx2,hyper,integral_idx=0):
        """
        estimate the \int \int k(x1,x2) dx1 dx2 at dimension integral_idx
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        temp1=np.atleast_2d(xx1[:,integral_idx]).T
        temp2=np.atleast_2d(xx2[:,integral_idx]).T
        
        Euc_dist_xint_xint=euclidean_distances(temp1,temp2)
        #K_xint_x=variance*np.exp(-np.square(Euc_dist_xint_x)/lengthscale)
        
        return 0.5*np.sqrt(3.14)*lengthscale*variance*erf(Euc_dist_xint_xint/np.sqrt(lengthscale))
    
    
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
    
    def log_llk_drv(self,X,y,lengthscale,noise_delta=1e-5,noise_drv=1):

        hyper={}
        hyper['var']=1
        hyper['lengthscale']=lengthscale
        
        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*noise_delta     
        KK_x_v=self.cov_RBF_drv(self.Xdrv,X,hyper,self.drv_index).T
 
        try:
            KK_v_v=self.cov_RBF_drv_drv_itself(self.Xdrv, hyper,self.drv_index)
            +np.eye(self.Xdrv.shape[0])*noise_drv
            top_matrix=np.hstack((KK_x_x,KK_x_v))
            bottom_matrix=np.hstack((KK_x_v.T, KK_v_v))
            KK_combined_drv=np.vstack((top_matrix,bottom_matrix))
    
            Ldrv=scipy.linalg.cholesky(KK_combined_drv,lower=True)
            alpha=np.linalg.solve(KK_combined_drv,self.Y_combined)

        except: # singular
            #print("singular",hyper['lengthscale'],noise_delta,noise_drv)
            return -np.inf
        try:
            #print("okay",hyper['lengthscale'],noise_delta,noise_drv)
            first_term=-0.5*np.dot(self.Y_combined.T,alpha)
            W_logdet=np.sum(np.log(np.diag(Ldrv)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)
        #print(np.asscalar(logmarginal))

        return np.asscalar(logmarginal)
    
    def select_next_point_byentropy(self):
        """
        Select next point to evaluate by entropy reduction
        x_t
        """
        
        def acq_var(Xtest):
            if len(Xtest.shape)==1: # 1d
                Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
            #Xtest=self.Xscaler.transform(Xtest)
            KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
            KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)
    
            v=np.linalg.solve(self.L,KK_xTest_x.T)
            var=KK_xTest_xTest-np.dot(v.T,v)
    
            std=np.reshape(np.diag(var),(-1,1))
            return std
        
        def acq_var_drv(Xtest):
            if len(Xtest.shape)==1: # 1d
                Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
                
            #Xtest=self.Xscaler.transform(Xtest)
            KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
            KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)
            KK_xTest_xdrv=self.cov_RBF_drv(self.Xdrv,Xtest,self.hyper,self.drv_index).T
            KK_xTest_x_xdrv=np.hstack((KK_xTest_x,KK_xTest_xdrv))
            
            v=np.linalg.solve(self.Ldrv,KK_xTest_x_xdrv.T)
            var=KK_xTest_xTest-np.dot(v.T,v)
            std=np.reshape(np.diag(var),(-1,1))

            return std

            
        opts ={'maxiter':500,'maxfun':500,'disp': False}
        SearchSpaceScale=np.asarray([[0,1]])
        
        init_points = np.random.uniform(SearchSpaceScale[:,0], SearchSpaceScale[:,1],size=(5, 1))
        init_output=acq_var(init_points)
            
        x0=init_points[np.argmax(init_output)]
        res = minimize(lambda xx: -acq_var_drv(xx),x0,
                       bounds=SearchSpaceScale,method="L-BFGS-B",options=opts)
        
        max_var=acq_var_drv(res.x)
        return res.x,max_var
    
    def auto_train_bq(self,myfunction,MaxK=7):
        flagNonStop=True
        all_new_points= np.empty((0,self.dim), float)
        while(flagNonStop and len(self.Y_ori)<MaxK): # querying more points
            
            new_point,max_var=self.select_next_point_byentropy()
            if self.verbose:
                print("max_var",max_var)
            all_new_points=np.vstack((all_new_points,new_point))
            if max_var<1e-4: # this is a threshold to stop querying more points
                flagNonStop=False
                continue
                
            new_point=np.reshape(new_point,(1,1))
            new_point_ori=self.Xscaler.inverse_transform(new_point)
            output=myfunction(np.atleast_2d(new_point_ori))
            self.X_ori=np.vstack((self.X_ori,new_point_ori))
            output=output.data.cpu().numpy()
            self.Y_ori=np.vstack((self.Y_ori,output))
            
            self.fit_drv(self.X_ori,self.Y_ori,self.Xdrv_ori,self.Ydrv_ori,drv_index=0)

        return self.X_ori
    
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
    
    def optimise_drv(self):
        """
        Optimise the GP kernel hyperparameters
        Returns
        x_t
        """
        
        opts ={'maxiter':200,'maxfun':200,'disp': False}

        #bounds=np.asarray([[1,1],[0.0001,1]])
        bounds=np.asarray([[0.3,1.5]])
        
        init_theta = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20,1))
        logllk=[0]*init_theta.shape[0]
        for ii,val in enumerate(init_theta):      
            
            logllk[ii]=self.log_llk_drv(self.X,self.Y,val,
                  noise_delta=self.noise_delta,noise_drv=self.noise_drv)
            
        x0=init_theta[np.argmax(logllk)]

        res = minimize(lambda x: -self.log_llk_drv(self.X,self.Y,x,noise_delta=self.noise_delta,
                   noise_drv=self.noise_drv),x0,bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B
        
        if self.verbose:
            print("estimated hyperparameters:",res.x)
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
    
    def predict_gradient_with_drv(self,Xtest):
        """
        Predicting the gradient value at Xtest
        """
        Xtest=self.Xscaler.transform(Xtest)
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.cov_RBF_drv_drv_itself(Xtest,self.hyper,drv_index=0)+np.eye(Xtest.shape[0])*self.noise_drv
        #KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta

        KK_xTest_x=self.cov_RBF_drv(Xtest,self.X,self.hyper)
        
        KK_xTest_xdrv=self.cov_RBF_drv_drv(Xtest,self.Xdrv,self.hyper)
        #KK_xTest_xdrv=self.mycov(Xtest,self.Xdrv,self.hyper)

        KK_xTest_x_xdrv=np.hstack((KK_xTest_x,KK_xTest_xdrv))

        mean_grad=np.dot(KK_xTest_x_xdrv,self.alphadrv)
        v=np.linalg.solve(self.Ldrv,KK_xTest_x_xdrv.T)
        var=KK_xTest_xTest-np.dot(v.T,v)


        mean_grad_ori=    mean_grad*np.std(self.Y_ori)+np.mean(self.Y_ori)
        std_grad=np.reshape(np.diag(var),(-1,1))
        
        std_grad_ori=std_grad*np.std(self.Y_ori)#+np.mean(self.Y_ori)
        
        return mean_grad,std_grad,mean_grad_ori,std_grad_ori
        
    def predict_integral(self,Xtest):
        """
        Given the original function X,y
        Given some points of the integral function Xint, yint
        predict the integral function at Xtest

        This is equivalent to given the derivative Xdrv, ydrv
        Given some points of the original func X, y
        predict the original function at Xtest
        """
        integral_idx=0
        Xtest=self.Xscaler.transform(Xtest)
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.cov_RBF_integral_integral(Xtest,Xtest,self.hyper,integral_idx=0) \
                                +np.eye(Xtest.shape[0])*self.noise_drv*0
        #KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta

        # integral->integral or original->original
        KK_xTest_x=self.cov_RBF_integral_integral(Xtest,self.X,self.hyper,integral_idx)
        
        # integral -> original or original -> dev
        KK_xTest_xdrv=self.cov_RBF_integral(Xtest,self.Xdrv,self.hyper,integral_idx)
        #KK_xTest_xdrv=self.mycov(Xtest,self.Xdrv,self.hyper)

        KK_xTest_x_xdrv=np.hstack((KK_xTest_x,KK_xTest_xdrv))
        
        KK_x_x=self.cov_RBF_integral_integral(self.X,self.X,self.hyper)+np.eye(len(self.X))*self.noise_delta*1 
        KK_x_v=self.cov_RBF_integral(self.Xdrv,self.X,self.hyper,integral_idx).T

        KK_v_v=self.mycov(self.Xdrv, self.Xdrv, self.hyper)\
                    +np.eye(self.Xdrv.shape[0])*self.noise_drv*0
        top_matrix=np.hstack((KK_x_x,KK_x_v))
        bottom_matrix=np.hstack((KK_x_v.T, KK_v_v))
        self.KK_combined_drv=np.vstack((top_matrix,bottom_matrix))

        self.Ldrv=scipy.linalg.cholesky(self.KK_combined_drv,lower=True)
                

        # [y, ydrv]
        #mean_grad=np.dot(KK_xTest_x_xdrv,self.alphadrv)
        v=np.linalg.solve(self.Ldrv,KK_xTest_x_xdrv.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        mean_grad_ori=    mean_grad*np.std(self.Y_ori)+np.mean(self.Y_ori)
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
    
    def find_key_point(self):
        # finding a single point giving the value = the integral value
        #integral_val=self.integral()[1]
        integral_val=self.integral_by_GPmean()[1]
        #desired_score=2*integral_val-np.max(self.Y_ori)
        desired_score=integral_val
        if self.verbose:
            print("integral_val",integral_val)
        
        def acq_mean(Xtest):
            if len(Xtest.shape)==1: # 1d
                Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
                
            KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)
            KK_xTest_xdrv=self.cov_RBF_drv(self.Xdrv,Xtest,self.hyper,self.drv_index).T
            KK_xTest_x_xdrv=np.hstack((KK_xTest_x,KK_xTest_xdrv))
            mean=np.dot(KK_xTest_x_xdrv,self.alphadrv)
            mean_ori=mean*np.std(self.Y_ori)+np.mean(self.Y_ori)
            return np.abs(desired_score-mean_ori)

        opts ={'maxiter':100,'maxfun':100,'disp': False}
        SearchSpaceScale=np.asarray([[0,1]])
        
        init_points = np.random.uniform(SearchSpaceScale[:,0], SearchSpaceScale[:,1],size=(5, 1))
        init_output=acq_mean(init_points)
            
        x0=init_points[np.argmin(init_output)]
        res = minimize(lambda xx: acq_mean(xx),x0,
                       bounds=SearchSpaceScale,method="L-BFGS-B",options=opts)
        
        gap_at_keypoint=acq_mean(res.x)
        return res.x,gap_at_keypoint
    
    def integral_by_GPmean(self):
        
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        
        # using Cholesky update
        pred_mean,pred_var,pred_mean_ori,pred_var_ori=self.predict_with_drv(Xtest)
        return np.mean(pred_mean),np.mean(pred_mean_ori)        
        
    def integral(self):
        # compute the integral using closed-form
        
        ubs=self.SearchSpace[:,1]
        lbs=self.SearchSpace[:,0]
        sqrt_ls=np.sqrt(self.hyper['lengthscale'])
        term1 = np.sqrt(3.14)*self.hyper['var']*sqrt_ls/(2*(ubs-lbs))
        term2= erf((self.X_ori-lbs)/sqrt_ls)-erf((self.X_ori-ubs)/sqrt_ls)
        closed_form_integral=np.dot(term1*term2.T,self.alpha)
        closed_form_integral_ori=closed_form_integral*np.std(self.Y_ori)+np.mean(self.Y_ori)
        return np.asscalar(closed_form_integral),np.asscalar(closed_form_integral_ori)
    
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
        
    def plot_integral_1d(self,myfunction):
        # given gradient func, and a few points in original function
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        pred_int,pred_std_of_int,pred_int_ori,pred_std_of_int_ori=self.predict_integral(Xtest)

        fig = plt.figure(figsize=(8,5))
        
        ax = fig.add_subplot(1,1,1)
        
        #my_cmap = plt.get_cmap('Blues')
        CS_acq=ax.plot(Xtest,pred_int.reshape(Xtest.shape),':k',linewidth=3,label='Integral of Derivative')
        
        temp_xaxis=np.concatenate([Xtest, Xtest[::-1]])
        temp_yaxis=np.concatenate([pred_int - 1.9600 * pred_std_of_int, (pred_int + 1.9600 * pred_std_of_int)[::-1]])
        ax.scatter(self.X,self.Y,marker='s',s=100,color='g',label='Obs')  
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        #ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')

        #y_org=myfunction(Xtest)
        
        #CS_acq=ax.plot(Xtest,y_org.reshape(Xtest.shape),'r',label="True Function")
        
        ax.legend(prop={'size': 14})
        
        ax.set_ylabel("$f(x)$",fontsize=16)
        ax.set_xlabel("Beta",fontsize=16)
        
        ax.set_title("Integral Function $\int f'$ Given f'",fontsize=20)  
        #strPath="plot/MNIST_TVO_VAE_BQD_{:d}_points.pdf".format(len(self.Y))
        #fig.savefig("plot/der.pdf")
        
    def plot_gradient_1d_drv(self,myfunction):
        # plot 1d function using derivative observations
        Xtest = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 100)
        Xtest=np.atleast_2d(Xtest).T
        pred_grad,pred_std_of_grad,pred_grad_ori,pred_std_of_grad_ori=self.predict_gradient_with_drv(Xtest)

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