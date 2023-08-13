### This text describes the supplementary material and official implementation of		   ###
##     Doubly Robust Interval Estimation for Optimal Policy Evaluation in Online Learning       ##

###  Authors ### 
Ye Shen*, North Carolina State University;
Hengrui Cai*, University of California Irvine;
Rui Song, Amazon

*Equal Contribution

###  Abstract ### 
Evaluating the performance of an ongoing policy plays a vital role in many areas such as medicine and economics, to provide crucial instructions on the early-stop of the online experiment and timely feedback from the environment. Policy evaluation in online learning thus attracts increasing attention by inferring the mean outcome of the optimal policy (i.e., the value) in real-time. Yet, such a problem is particularly challenging due to the dependent data generated in the online environment, the unknown optimal policy, and the complex exploration and exploitation trade-off in the adaptive experiment. In this paper, we aim to overcome these difficulties in policy evaluation for online learning. We explicitly derive the probability of exploration that quantifies the probability of exploring non-optimal actions under commonly used bandit algorithms. We use this probability to conduct valid inference on the online conditional mean estimator under each action and develop the **d**oubly **r**obust int**e**rv**a**l esti**m**ation (DREAM) method to infer the value under the estimated optimal policy in online learning. The proposed value estimator provides double protection for consistency and is asymptotically normal with a Wald-type confidence interval provided. Extensive simulation studies and real data applications are conducted to demonstrate the empirical validity of the proposed DREAM method.
### Full paper  ###
https://arxiv.org/abs/2110.15501

###  Requirements  ### 

 - Python 3.7
 - `numpy`
 - `pandas`
 - `sklearn`
 - `math`
 - `scipy`
 - `typing`
 - `multiprocess`
 - `tqdm`
 - `functools`

###  Contents ### 

  1. `README.txt`: implementation details of source code and contents

  2. `Plots`: plots for simulation results and real data application

  3. `Codes and Datasets`: Source code of DREAM and data used for real data application

     a). `Contextual.py`: main function for algorithm DREAM.

     b). `Simulation.py`: main codes for simulation experiments
     
     c). `Real Data`: datasets and codes for real data applications, including the main function `Realdata.py` and implementation codes `RealdataApplication.py`.

     d). `Plot.py`: codes for generating plots, including simulation studies and real data analysis.

See more details of numerical studies in Section 5 and Section 6 of our main paper.   



