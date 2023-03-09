#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:43:28 2021

"""
import numpy as np
import math
from scipy.integrate import dblquad
import matplotlib.pyplot as plt


def org_rep_res(rep_res):
    rep_res = np.array(rep_res)
    PEXPLOR = rep_res[:,0,:]
    Vhat = rep_res[:,1,:]
    Sigmav = rep_res[:,2,:]
    Coverv = rep_res[:,3,:]
    R_mean = rep_res[:,4,:]
    R_std = rep_res[:,5,:]
    Coverv_mean  = rep_res[:,6,:]
    return PEXPLOR,Vhat,Sigmav,Coverv,R_mean,R_std, Coverv_mean
    

def plot_bias(method,T0, T, trueV, Vhat, ridge_Vhat,phat_Vhat,all_Vhat,R_mean, front_size,T00=0,
              ax = None):
    Vhat_bias =  Vhat.mean(axis=0) - trueV
    ridge_Vhat_bias =  ridge_Vhat.mean(axis=0) - trueV
    phat_Vhat_bias =  phat_Vhat.mean(axis=0) - trueV
    R_mean_bias =  R_mean.mean(axis=0) - trueV
    if ax is None:
        plt.hlines(0, T0,T, colors = "r")
        plt.plot(range(T0+T00,T),Vhat_bias[T00:T-T0])  
        plt.plot(range(T0+T00,T),ridge_Vhat_bias[T00:T-T0])
        plt.plot(range(T0+T00,T),phat_Vhat_bias[T00:T-T0]) 
        plt.plot(range(T0+T00,T),R_mean_bias[T00:T-T0])
        plt.xlabel("Time")
        plt.ylabel("Bias")
    else:
        ax.hlines(0, T0,T, colors = "r")
        ax.plot(range(T0+T00,T),Vhat_bias[T00:T-T0])  
        ax.plot(range(T0+T00,T),ridge_Vhat_bias[T00:T-T0])
        ax.plot(range(T0+T00,T),phat_Vhat_bias[T00:T-T0]) 
        ax.plot(range(T0+T00,T),R_mean_bias[T00:T-T0])
        plt.xlabel("Time",fontsize= front_size)
        plt.ylabel("Bias",fontsize= front_size)
        

    
def plot_CPv(method,T0, T, Coverv, ridge_Coverv,phat_Coverv,all_Coverv,Coverv_mean, front_size,T00=0,
             ax = None):
    CPv =  Coverv.mean(axis=0)
    ridge_CPv =  ridge_Coverv.mean(axis=0)
    phat_CPv =  phat_Coverv.mean(axis=0)
    CPv_mean =  Coverv_mean.mean(axis=0)
    plt.hlines(0.95, T0,T, colors = "r")
    if ax is None:
        plt.plot(range(T0+T00,T),CPv[T00:T-T0])
        plt.plot(range(T0+T00,T),ridge_CPv[T00:T-T0])
        plt.plot(range(T0+T00,T),phat_CPv[T00:T-T0])
        plt.plot(range(T0+T00,T),CPv_mean[T00:T-T0])
        plt.xlabel("Time",fontsize=front_size)
        plt.ylabel("Coverage Probability",fontsize=front_size)
    else:
        ax.plot(range(T0+T00,T),CPv[T00:T-T0])
        ax.plot(range(T0+T00,T),ridge_CPv[T00:T-T0])
        ax.plot(range(T0+T00,T),phat_CPv[T00:T-T0])
        ax.plot(range(T0+T00,T),CPv_mean[T00:T-T0])
        plt.xlabel("Time",fontsize=front_size)
        plt.ylabel("Coverage Probability",fontsize=front_size)
        


def std_ratio(Sigmav,Vhat):
    SigmavMean = Sigmav.mean(axis=0)
    VStd = Vhat.std(axis=0)
    ratio = SigmavMean/VStd
    return ratio

def plot_ratio(method,T0, T,Vhat,ridge_Vhat,phat_Vhat,all_Vhat,R_mean, Sigmav,
               ridge_Sigmav,phat_Sigmav,all_Sigmav,R_std, front_size,T00=0,
               ax=None):
    ratio = std_ratio(Sigmav,Vhat)
    ridge_ratio = std_ratio(ridge_Sigmav,ridge_Vhat)
    phat_ratio = std_ratio(phat_Sigmav,phat_Vhat)
    R_mean_ratio = std_ratio(R_std,R_mean)
    if ax is None:
        plt.hlines(1, T0+T00,T, colors = "r")
        plt.plot(range(T0+T00,T),ratio[T00:T-T0])
        plt.plot(range(T0+T00,T),ridge_ratio[T00:T-T0])
        plt.plot(range(T0+T00,T),phat_ratio[T00:T-T0])
        plt.plot(range(T0+T00,T),R_mean_ratio[T00:T-T0])
        plt.xlabel("Time",fontsize=front_size)
        plt.ylabel("SE/MC Std ",fontsize=front_size)
    else:
        ax.hlines(1, T0+T00,T, colors = "r")
        ax.plot(range(T0+T00,T),ratio[T00:T-T0])
        ax.plot(range(T0+T00,T),ridge_ratio[T00:T-T0])
        ax.plot(range(T0+T00,T),phat_ratio[T00:T-T0])
        ax.plot(range(T0+T00,T),R_mean_ratio[T00:T-T0])
        plt.xlabel("Time",fontsize=front_size)
        plt.ylabel("SE/MC Std ",fontsize=front_size)
    


## Demonstration for Probability of Exploration

beta0 = np.array([2, -1, 1.5])
beta1 = np.array([1, 4, -1.5])
s0 = 1
s1 = 1
x1_range = (0,2*math.pi)
x2_range = (0,2*math.pi)
mu0 = lambda x: np.dot(beta0,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
mu1 = lambda x: np.dot(beta1,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
trueV= dblquad(lambda x,y: max(mu0(np.array([1,x,y])),mu1(np.array([1,x,y])))/
               (x1_range[1]-x1_range[0])/(x2_range[1]-x2_range[0]) , 
               x1_range[0],x1_range[1], x2_range[0],x2_range[1])[0]

T0 = 30  
T = 210 
B= 50000 


UCB_rep_res_1 = np.load('Simulation_results/Contextual_PE/UCB_rep_res_1.npy')
UCB_rep_res_2 = np.load('Simulation_results/Contextual_PE/UCB_rep_res_2.npy')
TS_rep_res_1 = np.load('Simulation_results/Contextual_PE/TS_rep_res_1.npy')
TS_rep_res_05 = np.load('Simulation_results/Contextual_PE/TS_rep_res_05.npy')
EG_rep_res = np.load('Simulation_results/Contextual_PE/EG_rep_res.npy')



UCB_PEXPLOR_1 = org_rep_res(UCB_rep_res_1)[0].mean(axis=0)
UCB_PEXPLOR_2 = org_rep_res(UCB_rep_res_2)[0].mean(axis=0)
TS_PEXPLOR_1 = org_rep_res(TS_rep_res_1)[0].mean(axis=0)
TS_PEXPLOR_05 = org_rep_res(TS_rep_res_05)[0].mean(axis=0)
EG_PEXPLOR =org_rep_res(EG_rep_res)[0].mean(axis=0)



UCB_logPEXPLOR_1 =np.log(UCB_PEXPLOR_1)
UCB_logPEXPLOR_2 = np.log(UCB_PEXPLOR_2)
TS_logPEXPLOR_1 =np.log(TS_PEXPLOR_1)
TS_logPEXPLOR_05 =np.log(TS_PEXPLOR_05)
EG_logPEXPLOR = np.log(EG_PEXPLOR)


fig = plt.figure(figsize = (15,12))

T00 = 30
T1=200

# plt.plot(np.sqrt(range(T0+T00,T1)),UCB_logPEXPLOR_1[T00:T1-T])
# plt.plot(np.sqrt(range(T0+T00,T1)),UCB_logPEXPLOR_2[T00:T1-T])
# plt.plot(np.sqrt(range(T0+T00,T1)),TS_logPEXPLOR_05[T00:T1-T])
# plt.plot(np.sqrt(range(T0+T00,T1)),TS_logPEXPLOR_1[T00:T1-T])
# plt.plot(np.sqrt(range(T0+T00,T1)),EG_logPEXPLOR[T00:T1-T])

mylinewidth=2.5
plt.plot(np.array(range(T0+T00,T1)),UCB_logPEXPLOR_1[T00:T1-T],linewidth=mylinewidth)
plt.plot(np.array(range(T0+T00,T1)),UCB_logPEXPLOR_2[T00:T1-T],linewidth=mylinewidth)
plt.plot(np.array(range(T0+T00,T1)),TS_logPEXPLOR_05[T00:T1-T],linewidth=mylinewidth)
plt.plot(np.array(range(T0+T00,T1)),TS_logPEXPLOR_1[T00:T1-T],linewidth=mylinewidth)
plt.plot(np.array(range(T0+T00,T1)),EG_logPEXPLOR[T00:T1-T],linewidth=mylinewidth)

myfrontsize = 23
plt.xticks (fontsize=myfrontsize)
plt.yticks (fontsize=myfrontsize)
plt.rcParams["font.family"] = "Times New Roman"
plt.legend([r"UCB,$c_t$=1",r"UCB,$c_t$=2",r'TS, $\rho$=0.5',r'TS, $\rho$=1',
            r'EG, $\epsilon_t = 0.1(t+1)^{-0.4}$'], fontsize=myfrontsize-4,
           loc = (-0.05, 1.05),ncol=5)
plt.xlabel(r"$t$",fontsize=myfrontsize )
plt.ylabel("log Probability of Exploration",fontsize=myfrontsize)
fig.savefig('Plots/PE.png', bbox_inches='tight')




##  Coverage Probabilities under DREAM


T0 = 50    
T = 2000 
B= 1000 
beta0 = np.array([2, -1, 1.5])
beta1 = np.array([1, 4, -1.5])
s0 = 0.5
s1 = 0.5
x1_range = (0,2*math.pi)
x2_range = (0,2*math.pi)
mu0 = lambda x: np.dot(beta0,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
mu1 = lambda x: np.dot(beta1,np.hstack((x[0],math.cos(x[1]),math.cos(x[2]))))
trueV= dblquad(lambda x,y: max(mu0(np.array([1,x,y])),mu1(np.array([1,x,y])))/
               (x1_range[1]-x1_range[0])/(x2_range[1]-x2_range[0]) , 
               x1_range[0],x1_range[1], x2_range[0],x2_range[1])[0]



UCB_rep_res = np.load('Simulation_results/Contextual_CPV/UCB_rep_res.npy')
UCB_phat_rep_res = np.load('Simulation_results/Contextual_CPV/UCB_phat_rep_res.npy')
UCB_ridge_rep_res = np.load('Simulation_results/Contextual_CPV/UCB_ridge_rep_res.npy')
UCB_all_rep_res = np.load('Simulation_results/Contextual_CPV/UCB_all_rep_res.npy')


fig = plt.figure(figsize = (16,4))
myfrontsize = 20

ax1 = fig.add_subplot(1,3,1)
plot_CPv("UCB, c=1",T0, T, org_rep_res(UCB_rep_res)[3], org_rep_res(
    UCB_ridge_rep_res)[3],org_rep_res(UCB_phat_rep_res)[3],
    org_rep_res(UCB_all_rep_res)[3], org_rep_res(UCB_rep_res)[6],T00=20,ax = ax1,front_size=myfrontsize)
ax1.xaxis.set_tick_params(labelsize=myfrontsize)
ax1.yaxis.set_tick_params(labelsize=myfrontsize)
ax1.set_yticks([0, 0.2,0.4,0.6,0.8,0.95])


ax2 = fig.add_subplot(1,3,2)
ax2.xaxis.set_tick_params(labelsize=myfrontsize)
ax2.yaxis.set_tick_params(labelsize=myfrontsize)
plot_bias( "UCB, c=1", T0 ,T,trueV, org_rep_res(UCB_rep_res)[1], 
          org_rep_res(UCB_ridge_rep_res)[1],org_rep_res(UCB_phat_rep_res)[1], 
       org_rep_res(UCB_all_rep_res)[1], org_rep_res(UCB_rep_res)[4],T00=20,ax = ax2,front_size=myfrontsize)


ax3 = fig.add_subplot(1,3,3)
ax3.xaxis.set_tick_params(labelsize=myfrontsize)
ax3.yaxis.set_tick_params(labelsize=myfrontsize)
plot_ratio("UCB, c=1",T0, T, 
        org_rep_res(UCB_rep_res)[1], org_rep_res(UCB_ridge_rep_res)[1],
        org_rep_res(UCB_phat_rep_res)[1], org_rep_res(UCB_all_rep_res)[1], 
        org_rep_res(UCB_rep_res)[4],
        org_rep_res(UCB_rep_res)[2], org_rep_res(UCB_ridge_rep_res)[2],
        org_rep_res(UCB_phat_rep_res)[2],org_rep_res(UCB_all_rep_res)[2], 
        org_rep_res(UCB_rep_res)[5],T00=20,ax = ax3,front_size=myfrontsize)

plt.tight_layout(pad=4)
fig.legend(["Corrected models",r"Misspecified $\mu$",r'Misspecified $\kappa_t$',
            'Average reward'], fontsize=myfrontsize,loc = (0.08, 0.86),ncol=5)

fig.savefig('Plots/Contextual_UCB.png', bbox_inches='tight')








TS_rep_res = np.load('Simulation_results/Contextual_CPV/TS_rep_res.npy')
TS_phat_rep_res = np.load('Simulation_results/Contextual_CPV/TS_phat_rep_res.npy')
TS_ridge_rep_res = np.load('Simulation_results/Contextual_CPV/TS_ridge_rep_res.npy')
TS_all_rep_res = np.load('Simulation_results/Contextual_CPV/TS_all_rep_res.npy')

fig = plt.figure(figsize = (16,4))
myfrontsize = 20

ax1 = fig.add_subplot(1,3,1)
plot_CPv("TS, rho=1",T0, T, org_rep_res(TS_rep_res)[3], 
         org_rep_res(TS_ridge_rep_res)[3],org_rep_res(TS_phat_rep_res)[3],
    org_rep_res(TS_all_rep_res)[3], org_rep_res(TS_rep_res)[6],T00=20,ax = ax1,front_size=myfrontsize)
ax1.xaxis.set_tick_params(labelsize=myfrontsize)
ax1.yaxis.set_tick_params(labelsize=myfrontsize)
ax1.set_yticks([0, 0.2,0.4,0.6,0.8,0.95])


ax2 = fig.add_subplot(1,3,2)
ax2.xaxis.set_tick_params(labelsize=myfrontsize)
ax2.yaxis.set_tick_params(labelsize=myfrontsize)
plot_bias( "TS, rho=1", T0 ,T,trueV, org_rep_res(TS_rep_res)[1], org_rep_res(
    TS_ridge_rep_res)[1],org_rep_res(TS_phat_rep_res)[1], org_rep_res(
        TS_all_rep_res)[1], org_rep_res(TS_rep_res)[4],T00=20,
          ax = ax2,front_size=myfrontsize)


ax3 = fig.add_subplot(1,3,3)
ax3.xaxis.set_tick_params(labelsize=myfrontsize)
ax3.yaxis.set_tick_params(labelsize=myfrontsize)
plot_ratio("TS, rho=1",T0, T, 
            org_rep_res(TS_rep_res)[1], org_rep_res(TS_ridge_rep_res)[1],
            org_rep_res(TS_phat_rep_res)[1], org_rep_res(TS_all_rep_res)[1], 
            org_rep_res(TS_rep_res)[4],
            org_rep_res(TS_rep_res)[2], org_rep_res(TS_ridge_rep_res)[2],
            org_rep_res(TS_phat_rep_res)[2],org_rep_res(TS_all_rep_res)[2], 
            org_rep_res(TS_rep_res)[5],T00=20,
            ax = ax3,front_size=myfrontsize)
plt.tight_layout(pad=4)
fig.legend(["Corrected models",r"Misspecified $\mu$",r'Misspecified $\kappa_t$',
            'Average reward'], fontsize=myfrontsize,
           loc = (0.08, 0.86),ncol=5)
fig.savefig('Plots/Contextual_TS.png', bbox_inches='tight')








EG_rep_res = np.load('Simulation_results/Contextual_CPV/EG_rep_res.npy')
EG_phat_rep_res = np.load('Simulation_results/Contextual_CPV/EG_phat_rep_res.npy')
EG_ridge_rep_res = np.load('Simulation_results/Contextual_CPV/EG_ridge_rep_res.npy')
EG_all_rep_res = np.load('Simulation_results/Contextual_CPV/EG_all_rep_res.npy')




fig = plt.figure(figsize = (16,4))
myfrontsize = 20


ax1 = fig.add_subplot(1,3,1)
ax1.xaxis.set_tick_params(labelsize=myfrontsize)
ax1.yaxis.set_tick_params(labelsize=myfrontsize)
plot_CPv("EG, epsilon = 0.5t^(-1/3)",T0, T, org_rep_res(EG_rep_res)[3], 
         org_rep_res(EG_ridge_rep_res)[3],org_rep_res(EG_phat_rep_res)[3],
         org_rep_res(EG_all_rep_res)[3], org_rep_res(EG_rep_res)[6],T00=20,
         ax = ax1,front_size=myfrontsize)
ax1.set_yticks([0, 0.2,0.4,0.6,0.8,0.95])


ax2 = fig.add_subplot(1,3,2)
ax2.xaxis.set_tick_params(labelsize=myfrontsize)
ax2.yaxis.set_tick_params(labelsize=myfrontsize)
plot_bias( "EG, epsilon = 0.5t^(-1/3)", T0 ,T,trueV, org_rep_res(EG_rep_res)[1], 
          org_rep_res(EG_ridge_rep_res)[1],org_rep_res(EG_phat_rep_res)[1], 
          org_rep_res(EG_all_rep_res)[1], org_rep_res(EG_rep_res)[4],T00=20,
          ax = ax2,front_size=myfrontsize)

ax3 = fig.add_subplot(1,3,3)
ax3.xaxis.set_tick_params(labelsize=myfrontsize)
ax3.yaxis.set_tick_params(labelsize=myfrontsize)
plot_ratio("EG, epsilon = 0.5t^(-1/3)",T0, T, 
            org_rep_res(EG_rep_res)[1], org_rep_res(EG_ridge_rep_res)[1],
            org_rep_res(EG_phat_rep_res)[1], org_rep_res(EG_all_rep_res)[1], 
            org_rep_res(EG_rep_res)[4],org_rep_res(EG_rep_res)[2], 
            org_rep_res(EG_ridge_rep_res)[2],org_rep_res(EG_phat_rep_res)[2],
            org_rep_res(EG_all_rep_res)[2], org_rep_res(EG_rep_res)[5],T00=20,
            ax = ax3,front_size=myfrontsize)
plt.tight_layout(pad=4)
fig.legend(["Corrected models",r"Misspecified $\mu$",r'Misspecified $\kappa_t$',
           'Average reward'], fontsize=myfrontsize,
           loc = (0.08, 0.86),ncol=5)

fig.savefig('Plots/Contextual_EG.png', bbox_inches='tight')






# Real data
T0 = 20
T = 200
sigma = 0.5 
B = 500


SEA_50_UCB_Coverv = np.load('Real Data/real_SEA_50/UCB' + '_Coverv.npy')
SEA_50_UCB_Coverv_mean = np.load('Real Data/real_SEA_50/UCB' + '_Coverv_mean.npy')
SEA_50_TS_Coverv = np.load('Real Data/real_SEA_50/TS' + '_Coverv.npy')
SEA_50_TS_Coverv_mean = np.load('Real Data/real_SEA_50/TS' + '_Coverv_mean.npy')
SEA_50_EG_Coverv = np.load('Real Data/real_SEA_50/EG' + '_Coverv.npy')
SEA_50_EG_Coverv_mean = np.load('Real Data/real_SEA_50/EG' + '_Coverv_mean.npy')

SEA_50000_UCB_Coverv = np.load('Real Data/real_SEA_50000/UCB' + '_Coverv.npy')
SEA_50000_UCB_Coverv_mean = np.load('Real Data/real_SEA_50000/UCB' + '_Coverv_mean.npy')
SEA_50000_TS_Coverv = np.load('Real Data/real_SEA_50000/TS' + '_Coverv.npy')
SEA_50000_TS_Coverv_mean = np.load('Real Data/real_SEA_50000/TS' + '_Coverv_mean.npy')
SEA_50000_EG_Coverv = np.load('Real Data/real_SEA_50000/EG' + '_Coverv.npy')
SEA_50000_EG_Coverv_mean = np.load('Real Data/real_SEA_50000/EG' + '_Coverv_mean.npy')



SEA_50_UCB_CPv  = SEA_50_UCB_Coverv[1:,:].mean(axis=0)
SEA_50_UCB_CPv_mean  = SEA_50_UCB_Coverv_mean[1:,:].mean(axis=0)
SEA_50_TS_CPv  = SEA_50_TS_Coverv[1:,:].mean(axis=0)
SEA_50_TS_CPv_mean  = SEA_50_TS_Coverv_mean[1:,:].mean(axis=0)
SEA_50_EG_CPv  = SEA_50_EG_Coverv[1:,:].mean(axis=0)
SEA_50_EG_CPv_mean  = SEA_50_EG_Coverv_mean[1:,:].mean(axis=0)
        


SEA_50000_UCB_CPv  = SEA_50000_UCB_Coverv[1:,:].mean(axis=0)
SEA_50000_UCB_CPv_mean  = SEA_50000_UCB_Coverv_mean[1:,:].mean(axis=0)
SEA_50000_TS_CPv  = SEA_50000_TS_Coverv[1:,:].mean(axis=0)
SEA_50000_TS_CPv_mean  = SEA_50000_TS_Coverv_mean[1:,:].mean(axis=0)
SEA_50000_EG_CPv  = SEA_50000_EG_Coverv[1:,:].mean(axis=0)
SEA_50000_EG_CPv_mean  = SEA_50000_EG_Coverv_mean[1:,:].mean(axis=0)


fig = plt.figure(figsize = (16,5))

ax1 = fig.add_subplot(1,2,1)
myfrontsize = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.xticks (fontsize=myfrontsize )
plt.yticks ( fontsize=myfrontsize )

plt.plot(range(T0,T),SEA_50_UCB_CPv)
plt.plot(range(T0,T),SEA_50_TS_CPv)
plt.plot(range(T0,T),SEA_50_EG_CPv)
plt.plot(range(T0,T),SEA_50_UCB_CPv_mean)
plt.plot(range(T0,T),SEA_50_TS_CPv_mean)
plt.plot(range(T0,T),SEA_50_EG_CPv_mean)
plt.hlines(0.95, T0,T, colors = "r")
plt.xlabel("Time",fontsize=myfrontsize)
plt.ylabel("Coverage Probability",fontsize=myfrontsize)

ax2 = fig.add_subplot(1,2,2)

myfrontsize = 20
# plt.rcParams["font.family"] = "Times New Roman"
plt.xticks (fontsize=myfrontsize )
plt.yticks ( fontsize=myfrontsize )

plt.plot(range(T0,T),SEA_50000_UCB_CPv)
plt.plot(range(T0,T),SEA_50000_TS_CPv)
plt.plot(range(T0,T),SEA_50000_EG_CPv)
plt.plot(range(T0,T),SEA_50000_UCB_CPv_mean)
plt.plot(range(T0,T),SEA_50000_TS_CPv_mean)
plt.plot(range(T0,T),SEA_50000_EG_CPv_mean)
plt.hlines(0.95, T0,T, colors = "r")
plt.xlabel("Time",fontsize=myfrontsize)
plt.ylabel("Coverage Probability",fontsize=myfrontsize)


fig.legend(["DREAM under UCB", "DREAM under TS","DREAM under EG",  "Average reward under UCB","Average reward under TS", "Average reward under EG"], 
           loc = (0.08, 0.80),ncol=3,fontsize=myfrontsize)
plt.tight_layout(pad=7)
fig.savefig('Plots/Real_data.png', bbox_inches='tight')






