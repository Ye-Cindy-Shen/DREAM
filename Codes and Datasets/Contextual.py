#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:23:51 2021

"""
import numpy as np
import math
from scipy.integrate import dblquad
from typing import Optional
from sklearn.linear_model import Ridge


class Contextual:
    def __init__(self, T0: int, T: int,  s0: float, s1: float, 
              beta0: np.array, beta1: np.array, x1_range: list, 
              x2_range: list, pt: Optional[float]=None) -> None:
        """
        This class Contextual is used to run Contextual Bandits
        
        Parameters
        ----------
        T0: int
            the length of Burning period
        T: int
            total decision time
        B: int
            Monte Carlo sample size
        beta0: np.array
            np.array of order 3
        beta1: np.array
            np.array of order 3
        s0: float
            the true standard deviation of arm 0
        s1: float
            the true standard deviation of arm 1
        x1_range: list
            the range of x1 is (x1_range[0],x1_range[1])
        x2_range: list
            the range of x2 is (x2_range[0],x2_range[1])
        pt: Optional[float]
            clipping rate
        
        Returns
        -------
        None
        
        """
        super(Contextual, self).__init__()
        self.T0 = T0
        self.T = T
        self.beta0 = beta0
        self.beta1 = beta1
        self.s0 = s0
        self.s1 = s1
        self.pt = pt
        self.x1_range = (x1_range[0],x1_range[1])
        self.x2_range = (x2_range[0],x2_range[1])
        self.trueV= dblquad(lambda x,y: max(self.mu0(np.array([1,x,y])),
                self.mu1(np.array([1,x,y])))/(x1_range[1]-x1_range[0])/
                            (x2_range[1]-x2_range[0]), x1_range[0],x1_range[1],
                            x2_range[0],x2_range[1])[0]
        return None
  
      
    def mu0(self,x:np.array) -> float:
        """
        This function is used to calculate the mean of arm 0
        
        Parameters
        ----------
        x: np.array
            np.array of order 3
            
        Returns
        -------
        float
        
        """
        res = np.dot(self.beta0,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
        return res


    
    def mu1(self,x:np.array) -> float:
       """
       This function is used to calculate the mean of arm 1
       
       Parameters
       ----------
       x: np.array
           np.array of order 3
           
       Returns
       -------
       float
       
       """
       res = np.dot(self.beta1,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
       return res
       


    def ridge_estimate(self,a: np.array,x: np.array,y: np.array,c: int,
                       predict_value: np.array, omega: Optional[int] =1,
                       misspecified: Optional[bool]= False) -> list:
        """
        This function is used to calculate ridge estimation
        
        Parameters
        ----------
        a: np.array
            array of action 
        x: np.array
            matrix of the independent variables
        y: np.array
            array of the dependent variable
        c: int
            target arm
        predic_value: np.array
            array of the independent varibales to be predict
        misspecified: Optional[bool]
            bool to indicate whether the regression model is misspecified
            
        Returns
        -------
        muhat: float
            estimated mean
        sigma_hat:float
            estimated standard deviation
        
        """        
        X = x[:,a==c]
        if X.shape[1] <= X.shape[0]:
            return None
        Y = y[a==c]
        if misspecified == False:
            X = np.vstack((X[0,:], np.cos(X[1,:]),np.cos(X[2,:])))
            if  predict_value.ndim ==1:
                newX = np.hstack((predict_value[0],
                                  np.cos(predict_value[1]),
                                  np.cos(predict_value[2])))
            else:
                newX = np.vstack((predict_value[0,:],
                                  np.cos(predict_value[1,:]),
                                  np.cos(predict_value[2,:])))
        if misspecified == True:
            X = np.vstack((X[0:], np.cos(X[1,:]),X[2,:]))
            if  predict_value.ndim ==1:
                newX= np.hstack((predict_value,
                                 np.cos(predict_value[1]),
                                        predict_value[2]))
            else:
                newX = np.vstack((predict_value[0,:],
                                  np.cos(predict_value[1,:]),
                                        predict_value[2,:]))
        
        
        clf = Ridge(alpha=omega,fit_intercept=False)
        clf.fit(X.T, Y)
        
        if  predict_value.ndim ==1:
            Ridge_inv = np.linalg.pinv(np.dot(X, X.T)+ omega*np.identity(X.shape[0]))
            sigma_hat = math.sqrt(np.dot(np.dot(newX.T,Ridge_inv),newX))
            beta_hat =  np.dot(Ridge_inv, np.dot(X, Y.T))
            muhat = np.dot(newX ,beta_hat)
        else:
            sigma_hat = -1
            muhat = clf.predict(newX.T)
        
        mu_hat = clf.predict(X.T)    
        residual = Y -  mu_hat
        sigma =  math.sqrt(np.linalg.norm(residual)**2/(len(Y) - X.shape[0]))
        
        return muhat,sigma_hat, sigma




    def bandits(self,method:str,ucb_c: Optional[float]=1, 
                ts_rho: Optional[float] = 1, mis_ridge: Optional[bool]=False,
                mis_phat: Optional[bool] = False): 
        """
        This function is used to run Contextual Bandits using one of the 
        following methods: Upper Confidence Bound (UCB), Thompson Sampling (TS) 
        and Epsilon Greedy (EG)
             
        
        Parameters
        ----------
        Method: str
            Upper Confidence Bound ("UCB"),  Thompson Sampling ("TS"), or 
            Epsilon Greedy ("EG")
        ucb_c: Optional[float] = 1
            parameter for Upper Confidence Bound (UCB)
        ts_rho: Optional[float] = 1
            parameter for Thompson Sampling (TS) 
        mis_ridge:  Optional[bool]
                bool to indicate whether the regression model is misspecified
        mis_phat: Optional[bool]
                bool to indicate whether the model for kappa is misspecified
            
        
        Returns
        -------
        PEXPLOR: np.array
            indicator of exploration
        Vhat: np.array
            estimated policy value under the estimated optimal policy
        Sigmav: np.array
            estimated variance of the estimated policy value under the 
            estimated optimal policy
        Coverv: np.array
            indicator of whether the estimated interval covers the true policy
            value
        Coverv_mean:np.array
            estimated optiaml policy value under reward mean
        R_mean:np.array
            estimated reward mean 
        R_std np.array
            estimated reward standard deviation
        
        """
        #initialization--
        X1 = np.random.uniform(self.x1_range[0],self.x1_range[1],size = self.T)
        X2 = np.random.uniform(self.x2_range[0],self.x2_range[1],size = self.T)
        X  = np.vstack((np.ones(self.T), X1,X2))
        Pihat = [int(self.mu1(X[:,t]) >= self.mu0(X[:,t]) ) for t in range(self.T0)]
        
        ## action
        A = np.random.binomial(1, 0.5, self.T0)
        
        ## reward
        sigma = [self.s0*(1-x)+self.s1*x for x in A]
        mean = [self.mu0(X[:,t])*(1-A[t])+self.mu1(X[:,t])*A[t] for t in range(self.T0)]
        R = np.random.normal(mean, sigma, size=self.T0) 
         
        Indicator = np.array(A==Pihat[:self.T0]).astype(int)
        Muhat0 = Muhat1  = Muhat =  Phat = PEXPLOR =  np.array([-1])    
        V = Vhat  = SigmaV = Coverv = np.array([-1])
        R_mean  = R_std = Coverv_mean =  np.array([-1])  
      

        fcover = lambda mu,sigma,c:int(((mu-1.96*sigma)<=c)and((mu+1.96*sigma)>=c)) 
        
        for t in range(self.T0,self.T):         
            muhat0,sigma0 = self.ridge_estimate(A,X[:,:t],R,0,X[:,t],mis_ridge)[0:2]
            muhat1,sigma1 = self.ridge_estimate(A,X[:,:t],R,1,X[:,t],mis_ridge)[0:2]
            Muhat0 = np.append(Muhat0,muhat0)
            Muhat1 = np.append(Muhat1,muhat1)
            
            muhat = max(muhat0, muhat1)
            Muhat = np.append(Muhat,muhat)
            
            pihat = int(muhat1 > muhat0)
            Pihat = np.append(Pihat,pihat)
            

            if method == "UCB":
                a=int( muhat1+ ucb_c *sigma1 >  muhat0 + ucb_c*sigma0) 
                 
            elif method == "EG":             
                epsilon_t = 0.1*(t+1)**(-1/2.5)
                delta  =  np.random.binomial(1, 1-epsilon_t/2, 1)
                a= delta*pihat +(1-delta)*np.random.binomial(1, 0.5, 1) 
                 
            elif method == "TS":              
                postsigma0=ts_rho*sigma0  
                postsigma1=ts_rho*sigma1
                postmuhat0 = np.random.normal(muhat0, postsigma0, size=1)
                postmuhat1 = np.random.normal(muhat1, postsigma1, size=1)
                a=int( postmuhat1 > postmuhat0) 
                 
            else:
                print(("The method should be one of the three: UCB / TS / EG"))

            A = np.append(A,a)
             
            # Clipping
            if self.pt is not None:
                tempX = X[:,:(t+1)]
                lambda_min = np.min(np.linalg.eig(np.dot(tempX, tempX.T))[0])
                tempX = tempX[:, A==(1-a)]
                sigma_hat_min = np.min(np.linalg.eig(np.dot(tempX, tempX.T))[0])
                if sigma_hat_min <  self.pt * lambda_min:
                    A[-1] = 1-a
            
            
            indicator = int(pihat == a)
            Indicator = np.append(Indicator,indicator)            
            
            # reward
            sigma = self.s0*(1-a)+self.s1*a
            mean = self.mu0(X[:,t])*(1-a)+ self.mu1(X[:,t])*a 
            r = np.random.normal(mean, sigma, size=1) 
            R = np.append(R,r)            

            # Probability of Exploration 
            Pexplore = 1-np.mean(Indicator)
            PEXPLOR = np.append(PEXPLOR,Pexplore)
            
            # Propensity score    
            phat = 1-Pexplore
            if mis_phat == True:
                phat = 0.5
            Phat = np.append(Phat,phat)
            
            # DREAM: AIPW for online optimization
            v = indicator/phat*(r - muhat)+ muhat
            V = np.append(V,v)
            
            vhat = np.mean(V[1:])
            Vhat =  np.append(Vhat,vhat)
             

            sigma0 = self.ridge_estimate(A[:t],X[:,:t],R[:t],0,X[:,t],mis_ridge)[2]
            sigma1 = self.ridge_estimate(A[:t],X[:,:t],R[:t],1,X[:,t],mis_ridge)[2]
            sigmav = (sigma0**2*(1-pihat)+ sigma1**2*pihat)/phat 
    
            Est_muhat0 =  self.ridge_estimate(A[:t],X[:,:t],R[:t],0,X[:,:t],mis_ridge)[0]
            Est_muhat1 =  self.ridge_estimate(A[:t],X[:,:t],R[:t],1,X[:,:t],mis_ridge)[0]
            Est_muhat = np.maximum(Est_muhat0,Est_muhat1)
            
            sigmaV = np.sqrt(sigmav+np.std(Est_muhat[1:])**2)/math.sqrt(len(Vhat[1:]))
            SigmaV = np.append(SigmaV,sigmaV)
            
            coverv = fcover(vhat,sigmaV,self.trueV) 
            Coverv = np.append(Coverv,coverv)
                   
            #Baseline
            R_mean = np.append(R_mean,np.mean(R))
            R_std  = np.append(R_std,np.std(R)/math.sqrt(len(Vhat[1:])))
            coverv_mean = fcover(np.mean(R),np.std(R)/math.sqrt(len(Vhat[1:])),self.trueV) 
            Coverv_mean = np.append(Coverv_mean,coverv_mean)
            
        return PEXPLOR[1:], Vhat[1:], SigmaV[1:], Coverv[1:], R_mean[1:], R_std[1:], Coverv_mean[1:]
