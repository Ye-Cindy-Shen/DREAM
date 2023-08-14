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

     A). `Contextual.py`: main functions for algorithm DREAM

     a)We build a class called `Contextual` to run Contextual Bandits. Parameters used for simulation setup are listed as follows:

	    - T0: int; the length of Burning period

     - T: int; total decision time

     - B: int; Monte Carlo sample size

     - beta0: np.array; np.array of order 3

     - beta1: np.array; np.array of order 3

     - s0: float; the true standard deviation of arm 0

     - s1: float; the true standard deviation of arm 1

     - x1_range: list; the range of x1 is (x1_range[0],x1_range[1])

     - x2_range: list; the range of x2 is (x2_range[0],x2_range[1])

     - pt: Optional[float]; clipping rate

     b) Within the class, we have two functions 'mu0(self,x:np.array)' and 'mu1(self,x:np.array)' to calculate the mean of the arms

     c) The function 'ridge_estimate' is used to calculate ridge estimation using inputs:

	    - a: np.array; array of action 

     - x: np.array; matrix of the independent variables
        
     - y: np.array; array of the dependent variable
        
     - c: int; target arm
        
     - predic_value: np.array; array of the independent varibales to be predict
    
       
     - misspecified: Optional[bool]; bool to indicate whether the regression model is misspecified

    
            
     And the function returns: 

     - muhat: float; estimated mean
        
     - sigma_hat:float; estimated standard deviation

	  d) The function 'bandits' is our main function to run  Contextual Bandits. The inputs are listed as follows:

     - Method: str; Upper Confidence Bound ("UCB"),  Thompson Sampling ("TS"), or Epsilon Greedy ("EG")
         
     -  ucb_c: Optional[float] = 1; parameter for Upper Confidence Bound (UCB)

     -  ts_rho: Optional[float] = 1; parameter for Thompson Sampling (TS) 

     - mis_ridge:  Optional[bool]; bool to indicate whether the regression model is misspecified

     - mis_phat: Optional[bool]; bool to indicate whether the model for kappa is misspecified
            
     And the outputs are:

     - PEXPLOR: np.array; indicator of exploration
        
     - Vhat: np.array; estimated policy value under the estimated optimal policy

     - Sigmav: np.array; estimated variance of the estimated policy value under the estimated optimal policy
        
     - Coverv: np.array; indicator of whether the estimated interval covers the true policy value
        
     - Coverv_mean:np.array; estimated optiaml policy value under reward mean
        
     - R_mean:np.array; estimated reward mean 
        
     - R_std np.array; estimated reward standard deviation

     B). `Simulation.py`: main codes for simulation experiments
     
     C). `Real Data`: dataset and codes for real data application, including main function `Realdata.py.py` and  main codes `RealdataApplication.py`.

     D). `Plot.py`: codes for generating plots, including simulation studies and real data analysis.

See more details of numerical studies in Section 5 and Section 6 of our main paper.   



