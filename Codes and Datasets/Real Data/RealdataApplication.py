#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:33:45 2021
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing 
from Realdata import Realdata



#SEA_50

SEA_50 = pd.read_csv('real_SEA_50/SEA_50.csv')
SEA_50.iloc[:,:-1] = preprocessing.scale(SEA_50.iloc[:,:-1])
SEA_50.iloc[:,-1] = pd.factorize(SEA_50.iloc[:,-1])[0].astype(np.uint16)

Real_SEA_50 = Realdata(T0 = 20, T = 200, B = 500, sigma = 0.5, df=SEA_50)

seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50.run_bandits(method= 'EG')

np.save('real_SEA_50/EG_Vhat.npy',Vhat)
np.save('real_SEA_50/EG_SigmaV.npy',SigmaV)
np.save('real_SEA_50/EG_Coverv.npy',Coverv) 

np.save('real_SEA_50/EG_Vipw.npy',Vipw)
np.save('real_SEA_50/EG_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50/EG_Coveripw.npy',Coveripw) 

np.save('real_SEA_50/EG_R_mean.npy',R_mean)
np.save('real_SEA_50/EG_R_std.npy',R_std)
np.save('real_SEA_50/EG_Coverv_mean.npy',Coverv_mean) 


seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50.run_bandits(method= 'UCB')


np.save('real_SEA_50/UCB_Vhat.npy',Vhat)
np.save('real_SEA_50/UCB_SigmaV.npy',SigmaV)
np.save('real_SEA_50/UCB_Coverv.npy',Coverv) 

np.save('real_SEA_50/UCB_Vipw.npy',Vipw)
np.save('real_SEA_50/UCB_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50/UCB_Coveripw.npy',Coveripw) 

np.save('real_SEA_50/UCB_R_mean.npy',R_mean)
np.save('real_SEA_50/UCB_R_std.npy',R_std)
np.save('real_SEA_50/UCB_Coverv_mean.npy',Coverv_mean) 


seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50.run_bandits(method= 'TS')


np.save('real_SEA_50/TS_Vhat.npy',Vhat)
np.save('real_SEA_50/TS_SigmaV.npy',SigmaV)
np.save('real_SEA_50/TS_Coverv.npy',Coverv) 

np.save('real_SEA_50/TS_Vipw.npy',Vipw)
np.save('real_SEA_50/TS_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50/TS_Coveripw.npy',Coveripw) 

np.save('real_SEA_50/TS_R_mean.npy',R_mean)
np.save('real_SEA_50/TS_R_std.npy',R_std)
np.save('real_SEA_50/TS_Coverv_mean.npy',Coverv_mean) 


#SEA_50000

SEA_50000= pd.read_csv('real_SEA_50000/SEA_50000.csv')
SEA_50000.iloc[:,:-1] = preprocessing.scale(SEA_50000.iloc[:,:-1])
SEA_50000.iloc[:,-1] = pd.factorize(SEA_50000.iloc[:,-1])[0].astype(np.uint16)

Real_SEA_50000= Realdata(T0 = 20, T = 200, B = 500, sigma = 0.5, df=SEA_50000)

seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50000.run_bandits(method= 'EG')

np.save('real_SEA_50000/EG_Vhat.npy',Vhat)
np.save('real_SEA_50000/EG_SigmaV.npy',SigmaV)
np.save('real_SEA_50000/EG_Coverv.npy',Coverv) 

np.save('real_SEA_50000/EG_Vipw.npy',Vipw)
np.save('real_SEA_50000/EG_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50000/EG_Coveripw.npy',Coveripw) 

np.save('real_SEA_50000/EG_R_mean.npy',R_mean)
np.save('real_SEA_50000/EG_R_std.npy',R_std)
np.save('real_SEA_50000/EG_Coverv_mean.npy',Coverv_mean) 


seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50000.run_bandits(method= 'UCB')


np.save('real_SEA_50000/UCB_Vhat.npy',Vhat)
np.save('real_SEA_50000/UCB_SigmaV.npy',SigmaV)
np.save('real_SEA_50000/UCB_Coverv.npy',Coverv) 

np.save('real_SEA_50000/UCB_Vipw.npy',Vipw)
np.save('real_SEA_50000/UCB_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50000/UCB_Coveripw.npy',Coveripw) 

np.save('real_SEA_50000/UCB_R_mean.npy',R_mean)
np.save('real_SEA_50000/UCB_R_std.npy',R_std)
np.save('real_SEA_50000/UCB_Coverv_mean.npy',Coverv_mean) 


seed=27695
np.random.seed(seed)
Vhat, SigmaV, Coverv,Vipw, Sigmaipw, Coveripw, R_mean, R_std, Coverv_mean = Real_SEA_50000.run_bandits(method= 'TS')


np.save('real_SEA_50000/TS_Vhat.npy',Vhat)
np.save('real_SEA_50000/TS_SigmaV.npy',SigmaV)
np.save('real_SEA_50000/TS_Coverv.npy',Coverv) 

np.save('real_SEA_50000/TS_Vipw.npy',Vipw)
np.save('real_SEA_50000/TS_Sigmaipw.npy',Sigmaipw)
np.save('real_SEA_50000/TS_Coveripw.npy',Coveripw) 

np.save('real_SEA_50000/TS_R_mean.npy',R_mean)
np.save('real_SEA_50000/TS_R_std.npy',R_std)
np.save('real_SEA_50000/TS_Coverv_mean.npy',Coverv_mean)
