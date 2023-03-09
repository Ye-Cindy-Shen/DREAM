#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:51:10 2021

"""

import numpy as np
import math
from typing import Optional
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge


class Realdata:
    def __init__(self,T0:int,T:int,B:int,sigma:float,df:pd.DataFrame,
                 pt: Optional[float]=None) -> None:
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
        sigma: float
            the standard deviation of reward
        df: pd.DataFrame
            dataset
            
        Returns
        -------
        None
        
        """
        super(Realdata, self).__init__()
        self.T0 = T0
        self.T = T
        self.B = B
        self.sigma = sigma
        self.df = df
        self.pt = pt
        return None
    
    
    def ridge_estimate(self,a: np.array,x: np.array,y: np.array,c: int,
                       predict_value: np.array, omega: Optional[int] =1) -> list:
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
        X = x[a==c]
        Y = y[a==c]
        if len(X) ==1:
            return None
        D = np.vstack((np.ones(len(X)),X.T))
        
        clf = Ridge(alpha=omega)
        clf.fit(X, Y)
        
        if  predict_value.ndim ==1:
            pred_D = np.append(1,predict_value)
            Ridge_inv = np.linalg.pinv(np.dot(D, D.T)+ omega*np.identity(np.size(D,0)))
            beta_hat =  np.dot(np.dot(Ridge_inv,D), Y)
            sigma_hat = math.sqrt(np.dot(np.dot(pred_D.T,Ridge_inv) ,pred_D))
            muhat = np.dot(pred_D,beta_hat)
        else:
            sigma_hat = -1
            muhat = clf.predict(predict_value)
        
        mu_hat = clf.predict(X)    
        residual = Y -  mu_hat
        sigma =  math.sqrt(np.linalg.norm(residual)**2/(len(Y) - X.shape[1]))
        
        return muhat,sigma_hat, sigma



    def bandits(self, Index:np.array, method: str, c: Optional[float]=2,
            rho: Optional[float] = 0.5)-> list: 
        """
        This function is used to run Contextual Bandits using one of the 
        following methods: Upper Confidence Bound (UCB), Thompson Sampling (TS) 
        and Epsilon Greedy (EG)
                  
        Parameters
        ----------
        Method: str
            Upper Confidence Bound ("UCB"),  Thompson Sampling ("TS"), or 
            Epsilon Greedy ("EG")
        Index:np.array
            sample from real data
        ucb_c: Optional[float] = 2
            parameter for Upper Confidence Bound (UCB)
        ts_rho: Optional[float] = 0.5
            parameter for Thompson Sampling (TS) 
          
        Returns
        -------
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
        
        #initialization
        A = np.random.binomial(1, 0.5, self.T0)
        Pihat = self.df.iloc[Index[:self.T0], -1]
        Indicator= np.array(self.df.iloc[Index[:self.T0], -1] == A).astype(int)
        R = np.random.normal(Indicator,self.sigma, size=self.T0)
         
        
        Muhat0 = Muhat1  = Muhat =  Phat = PEXPLOR =  np.array([-1])    
        V = Vhat  = SigmaV = Coverv = np.array([-1])
        R_mean  = R_std = Coverv_mean =  np.array([-1]) 
        Vipw = Sigmaipw = Coveripw = np.array([-1])  
      
        
        fcover = lambda mu,sigma,c: int( ((mu-1.96*sigma)<=c) and 
                                        ((mu+1.96*sigma)>=c) ) 
        
        
        for t in range(self.T0,self.T):                    
            muhat0 = self.ridge_estimate(A, self.df.iloc[Index[:t], :-1],R,0,
                                         self.df.iloc[Index[t], :-1])[0]
            muhat1 = self.ridge_estimate(A, self.df.iloc[Index[:t], :-1],R,1,
                                         self.df.iloc[Index[t], :-1])[0]
            Muhat0 = np.append(Muhat0,muhat0)
            Muhat1 = np.append(Muhat1,muhat1)
            
            muhat = max(muhat0, muhat1)
            Muhat = np.append(Muhat,muhat)
            
            pihat = int(muhat1 > muhat0)
            Pihat = np.append(Pihat,pihat)
            
            
            #===============================================================
            
            sigma0 = self.ridge_estimate(A, self.df.iloc[Index[:t], :-1],R,
                                             0, self.df.iloc[Index[t], :-1])[1]
            sigma1 = self.ridge_estimate(A, self.df.iloc[Index[:t], :-1],R,
                                             1, self.df.iloc[Index[t], :-1])[1]
                
               
            if method == "UCB":
                a=int( muhat1+c*sigma1 >  muhat0+c*sigma0) 
                 
            elif method == "EG":             
                epsilon_t = (t+1)**(-1/3)
                delta  =  np.random.binomial(1, 1-epsilon_t/2, 1)
                a= delta*pihat +(1-delta)*np.random.binomial(1, 0.5, 1) 
                 
            elif method == "TS":
                postsigma0=rho*sigma0  
                postsigma1=rho*sigma1
                 
                # sample from the posterior distribution
                postmuhat0 = np.random.normal(muhat0, postsigma0, size=1)
                postmuhat1 = np.random.normal(muhat1, postsigma1, size=1)
                 
                # action 
                a=int( postmuhat1 > postmuhat0) 
                 
            else:
                print(("The method should be one of the three: UCB / TS / EG"))
                return
             
            #===============================================================
            A = np.append(A,a)
             
            if self.pt is not None:
                tempX = self.df.iloc[Index[:(t+1)], :-1]
                lambda_min = np.min(np.linalg.eig(np.dot(tempX, tempX.T))[0])
                tempX = tempX[:, A==(1-a)]
                sigma_hat_min = np.min(np.linalg.eig(np.dot(tempX, tempX.T))[0])
                if sigma_hat_min <  self.pt * lambda_min:
                    A[-1] = 1-a            
                       
            # reward
            indicator = int(self.df.iloc[Index[t], -1] == a)
            Indicator = np.append(Indicator,indicator)
            r = np.random.normal(indicator,self.sigma, size=1)
            R = np.append(R,r)

           # Probability of Exploration 
            Pexplore = 1-np.mean(Indicator)
            PEXPLOR = np.append(PEXPLOR,Pexplore)
 
            # Propensity score    
            phat = 1-Pexplore
            Phat = np.append(Phat,phat)
            
            # DREAM: AIPW for online optimization
            v = indicator/phat*(r - muhat)+ muhat
            V = np.append(V,v)
            
            vhat = np.mean(V[1:])
            Vhat =  np.append(Vhat,vhat)
            
                                             
            sigma0 = self.ridge_estimate(A, self.df.iloc[Index[:(t+1)], :-1],R,
                                             0, self.df.iloc[Index[:(t+1)], :-1])[2]
            sigma1 = self.ridge_estimate(A, self.df.iloc[Index[:(t+1)], :-1],R,
                                             1, self.df.iloc[Index[:(t+1)], :-1])[2]
            
    
            sigmav = (sigma0**2*(1-pihat)+ sigma1**2*pihat)/phat         
            Est_muhat0 =  self.ridge_estimate(A, self.df.iloc[Index[:(t+1)], :-1],R,
                                             0, self.df.iloc[Index[:(t+1)], :-1])[0]
            Est_muhat1 =  self.ridge_estimate(A, self.df.iloc[Index[:(t+1)], :-1],R,
                                             1, self.df.iloc[Index[:(t+1)], :-1])[0]
            Est_muhat = np.maximum(Est_muhat0,Est_muhat1)
            
            sigmaV = np.sqrt(sigmav+np.std(Est_muhat[1:])**2)/math.sqrt(len(Vhat[1:]))
            SigmaV = np.append(SigmaV,sigmaV)
            
            coverv = fcover(vhat,sigmaV,1) 
            Coverv = np.append(Coverv,coverv)
            
            # IPW for online optimization
            vipw = indicator/phat*r
            Vipw = np.append(Vipw,vipw)

            
            sigmaipw = np.sqrt(np.mean(np.power(Vipw[1:]-vipw,2)))/math.sqrt(len(Vipw[1:]))
            Sigmaipw = np.append(Sigmaipw,sigmaipw )
            
            coveripw = fcover(vipw,sigmaipw,1) 
            Coveripw = np.append(Coveripw,coveripw)
            
            #Baseline
            R_mean = np.append(R_mean,np.mean(R))
            R_std  = np.append(R_std,np.std(R)/math.sqrt(len(Vhat[1:])))
            coverv_mean = fcover(np.mean(R),np.std(R)/math.sqrt(len(Vhat[1:])),1) 
            Coverv_mean = np.append(Coverv_mean,coverv_mean)
         
        return Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean

    
    
    def run_bandits(self,  method: str, c: Optional[float]=2,
                rho: Optional[float] = 0.5) -> list: 
        """
        This function is used to run Contextual Bandits using one of the 
        following methods: Upper Confidence Bound (UCB), Thompson Sampling (TS) 
        and Epsilon Greedy (EG)
                  
        Parameters
        ----------
        Method: str
            Upper Confidence Bound ("UCB"),  Thompson Sampling ("TS"), or 
            Epsilon Greedy ("EG")
        ucb_c: Optional[float] = 2
            parameter for Upper Confidence Bound (UCB)
        ts_rho: Optional[float] = 0.5
            parameter for Thompson Sampling (TS) 
          
        Returns
        -------
        Bias: np.array
            bias of estimated optimal policy value under DREAM
        Std: np.array
            estimated std of estimated optimal policy value under DREAM
        CPv: np.array
            coverage prabability of estimated optimal policy value under DREAM
        Ratio: np.array
            Monte Carlo standard error divided by estimated std of estimated 
            optimal policy value under DREAM
        Bias_mean: np.array
            bias of estimated optimal policy value under reward mean
        Std_mean: np.array
            estimated std of estimated optimal policy value under reward mean
        CPv_mean: np.array
            coverage prabability of estimated optimal policy value under 
            reward mean
        Ratio_mean: np.array
            Monte Carlo standard error divided by estimated std of estimated 
            optimal policy value under reward mean
         
        """
        Vhat = np.array([-1*np.ones(self.T-self.T0)])   
        SigmaV = np.array([-1*np.ones(self.T-self.T0)])   
        Coverv =  np.array([-1*np.ones(self.T-self.T0)]) 
        
        Vipw = np.array([-1*np.ones(self.T-self.T0)])   
        Sigmaipw = np.array([-1*np.ones(self.T-self.T0)])   
        Coveripw =  np.array([-1*np.ones(self.T-self.T0)])
        
        
        R_mean = np.array([-1*np.ones(self.T-self.T0)])   
        R_std = np.array([-1*np.ones(self.T-self.T0)])   
        Coverv_mean =  np.array([-1*np.ones(self.T-self.T0)])
        
        
       
        
        
        for i in tqdm(range(self.B)):
            Index = np.random.choice(a=len(self.df), size=self.T, 
                                     replace=False, p=None)
            vhat,sigmav,coverv, vipw, sigmaipw, coveripw,r_mean,r_std,coverv_mean = self.bandits(Index,
                                                                 method=method)           
            Vhat = np.vstack((Vhat,vhat[1:]))
            SigmaV = np.vstack((SigmaV,sigmav[1:]))
            Coverv = np.vstack((Coverv,coverv[1:]))
            
            Vipw = np.vstack((Vipw,vipw[1:]))
            Sigmaipw = np.vstack((Sigmaipw,sigmaipw[1:]))
            Coveripw = np.vstack((Coveripw,coveripw[1:]))
            
            
            R_mean = np.vstack((R_mean,r_mean[1:]))
            R_std = np.vstack((R_std,r_std[1:]))
            Coverv_mean = np.vstack((Coverv_mean,coverv_mean[1:]))

        
        return Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean
    
    
    