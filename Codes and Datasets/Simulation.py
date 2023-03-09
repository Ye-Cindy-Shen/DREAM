#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:07:55 2021

"""
import numpy as np
import multiprocess as mp
import math
from tqdm import tqdm
from functools import partial
from Contextual import Contextual




## Demonstration for Probability of Exploration

DREAM_PE = Contextual(T0 = 30 , T =  210, s0 = 0.1, s1 = 0.1, 
              beta0 = np.array([2, -1, 1.5]), beta1 = np.array([1, 4, -1.5]), 
              x1_range = (0,2*math.pi), x2_range = (0,2*math.pi),pt=0.05)

B= 50000
seed = 23333
np.random.seed(seed) # Random seed
seeds_list = np.random.randint(1, 1000000, size=B)

with mp.Pool(4) as pool:
    UCB_rep_res_1 = list(tqdm(pool.map(partial(DREAM_PE.bandits,"UCB", 1), 
                                       seeds_list), total=B))

with mp.Pool(4) as pool:
    UCB_rep_res_2 = list(tqdm(pool.map(partial(DREAM_PE.bandits,"UCB", 2), 
                                       seeds_list), total=B))

np.save('Simulation_results/Contextual_PE/UCB_rep_res_1.npy',UCB_rep_res_1)
np.save('Simulation_results/Contextual_PE/UCB_rep_res_2.npy',UCB_rep_res_2)


with mp.Pool(4) as pool:
    TS_rep_res_1 = list(tqdm(pool.map(partial(DREAM_PE.bandits,"TS", ts_rho=1), 
                                      seeds_list), total=B))

with mp.Pool(4) as pool:
    TS_rep_res_05 = list(tqdm(pool.map(partial(DREAM_PE.bandits,"TS",ts_rho=0.5),
                                       seeds_list), total=B))

np.save('Simulation_results/Contextual_PE/TS_rep_res_1.npy',TS_rep_res_1)
np.save('Simulation_results/Contextual_PE/TS_rep_res_05.npy',TS_rep_res_05)

with mp.Pool(4) as pool:
    EG_rep_res = list(tqdm(pool.map(partial(DREAM_PE.bandits, "EG"),
                                    seeds_list), total=B))

np.save('Simulation_results/Contextual_PE/EG_rep_res.npy',EG_rep_res)




##  Coverage Probabilities under DREAM


DREAM = Contextual(T0 = 50, T =  2000, s0 = 0.5, s1 = 0.5, 
              beta0 = np.array([2, -1, 1.5]), beta1 = np.array([1, 4, -1.5]), 
              x1_range = (0,2*math.pi), x2_range = (0,2*math.pi),pt=0.05)

B= 1000
seed = 23333
np.random.seed(seed) # Random seed
seeds_list = np.random.randint(1, 1000000, size=B)


with mp.Pool(4) as pool:
    UCB_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"UCB",1), 
                                     seeds_list), total=B))

with mp.Pool(4) as pool:
    UCB_phat_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"UCB",1, 
                                      mis_phat = True), seeds_list), total=B))
  
with mp.Pool(4) as pool:
    UCB_ridge_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"UCB",1, 
                                      mis_ridge = True), seeds_list), total=B))
 
with mp.Pool(4) as pool:
    UCB_all_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"UCB",1, 
                      mis_ridge =True,mis_phat =  True), seeds_list), total=B))

np.save('Simulation_results/Contextual_CPV/UCB_rep_res.npy',UCB_rep_res)
np.save('Simulation_results/Contextual_CPV/UCB_phat_rep_res.npy',UCB_phat_rep_res)
np.save('Simulation_results/Contextual_CPV/UCB_ridge_rep_res.npy',UCB_ridge_rep_res)
np.save('Simulation_results/Contextual_CPV/UCB_all_rep_res.npy',UCB_all_rep_res)





with mp.Pool(4) as pool:
    TS_rep_res = list(tqdm(pool.map(partial(DREAM.bandits, "TS", ts_rho=2),
                                    seeds_list), total=B))
    
with mp.Pool(4) as pool:
    TS_ridge_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"TS", ts_rho=2, 
                                    mis_ridge = True), seeds_list), total=B))
 
with mp.Pool(4) as pool:
    TS_phat_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"TS", ts_rho=2,
                                      mis_phat = True), seeds_list), total=B))
  
with mp.Pool(4) as pool:
    TS_all_rep_res = list(tqdm(pool.map(partial(DREAM.bandits,"TS", ts_rho=2, 
                     mis_ridge =True, mis_phat = True), seeds_list), total=B))
  
np.save('Simulation_results/Contextual_CPV/TS_rep_res.npy',TS_rep_res)
np.save('Simulation_results/Contextual_CPV/TS_phat_rep_res.npy',TS_phat_rep_res)
np.save('Simulation_results/Contextual_CPV/TS_ridge_rep_res.npy',TS_ridge_rep_res)
np.save('Simulation_results/Contextual_CPV/TS_all_rep_res.npy',TS_all_rep_res)




with mp.Pool(4) as pool:
    EG_rep_res = list(tqdm(pool.map(partial(DREAM.bandits, "EG"), seeds_list), 
                           total=B))

with mp.Pool(4) as pool:
    EG_ridge_rep_res = list(tqdm(pool.map(partial(DREAM.bandits, "EG",
                                      mis_phat = True), seeds_list), total=B))
 
with mp.Pool(4) as pool:
    EG_phat_rep_res = list(tqdm(pool.map(partial(DREAM.bandits, "EG", 
                                      mis_ridge = True), seeds_list), total=B))
  
with mp.Pool(4) as pool:
    EG_all_rep_res = list(tqdm(pool.map(partial(DREAM.bandits, "EG", 
                      mis_ridge = True,mis_phat = True), seeds_list), total=B))
  
np.save('Simulation_results/Contextual_CPV/EG_rep_res.npy',EG_rep_res)
np.save('Simulation_results/Contextual_CPV/EG_phat_rep_res.npy',EG_phat_rep_res)
np.save('Simulation_results/Contextual_CPV/EG_ridge_rep_res.npy',EG_ridge_rep_res)
np.save('Simulation_results/Contextual_CPV/EG_all_rep_res.npy',EG_all_rep_res)
