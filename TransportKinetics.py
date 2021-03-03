import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def t_test_rate(I,window,t_crit = - 1):
    
    # I (1darray): normalised intensity time trace
    # window (int): window for moving average of rate
    # t_crit (float): critical t value for d.o.fs = window -1 for t test
    
    # change in I in equal intervals is proportional to the rate
    
    deltaI = I[1:]-I[:-1]
    
    # variables stores whether decision has been made
    
    decision_made = False
    
    # T keeps track of start time of moving average window
    
    T = 0
    
    T_max = len(I)
    
    while not decision_made:
        #print('T = ',T)
        
        rate_chunk = deltaI[T:T+window]
        
        mean_rate = np.mean(rate_chunk)
        
        std_rate = np.std(rate_chunk,ddof =1) # use best estimate of standard deviation
        
        sem_rate = std_rate/np.sqrt(window)
        
        t = mean_rate/sem_rate
        print('latest t: ',t)
        if t < t_crit:
            
            decision_made = True
            
            T = T+3 # linearly interpolate to find best estimate of start of decrease
            
        elif T == T_max:
            # force decision
            
            decision_made = True
            
            
            
        else:
            
            T += 1 # if test fails increment T
            
    return 5*T # transform frame number to time in seconds


