import numpy
import math
import sys
import time
import random
import os

class solution:
    def __init__(self):
        self.thresholds=[]
        self.value=0

def otsu(Thresholds,file):
    Thresholds=Thresholds.tolist()
    Thresholds.append(256)
    Thresholds.insert(0, 0)
    Thresholds.sort()
    for i in range(len(Thresholds)):
        Thresholds[i]=math.floor(Thresholds[i])

    f = open(os.getcwd()+'\\greyscale_count\\'+file, "r")
    
    hist=[]    
    for line in range(256):
        hist.append(int(f.readline()))

    Total_Pixels = sum(hist)

    for i in range(len(hist)):                                              # Probabilities
        hist[i] = hist[i] / Total_Pixels


    cumulative_sum = []                                                     # declaractions
    cumulative_mean = []
    global_mean = 0
    Sigma = 0

    for i in range(len(Thresholds)-1):
        cumulative_sum.append(sum(hist[Thresholds[i]:Thresholds[i + 1]]))   # Cumulative sum of each Class

        cumulative = 0
        for j in range(Thresholds[i], Thresholds[i + 1]):
            cumulative = cumulative + (j + 1) * hist[j]
        try:   
            cumulative_mean.append(cumulative / (cumulative_sum[-1]+sys.float_info.epsilon))             # Cumulative mean of each Class
        except:
            cumulative_mean.append(cumulative)
        global_mean = global_mean + cumulative                              # Global Intensity Mean

    for i in range(len(cumulative_mean)):                                   # Computing Sigma
        Sigma = Sigma + (cumulative_sum[i] *
                        ((cumulative_mean[i] - global_mean) ** 2))

    return(Sigma)




def GWO(lb,ub,dim,SearchAgents_no,Max_iter,image):
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=0 #float("inf")
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=0 #float("inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=0 #float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        
    s=solution()

    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=otsu(Positions[i,:],image)
            
            # Update Alpha, Beta, and Delta
            if fitness>Alpha_score :
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=Alpha_score  # Update beta
                Beta_pos=Alpha_pos.copy()
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness<Alpha_score and fitness>Beta_score ):
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness<Alpha_score and fitness<Beta_score and fitness>Delta_score):                 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                

    Alpha_pos=Alpha_pos.tolist()
    for thresh in range(len(Alpha_pos)):
        Alpha_pos[thresh]=math.floor(Alpha_pos[thresh])
    s.thresholds=Alpha_pos  
    s.value=Alpha_score  
    return s


if __name__ == "__main__":

    runfile = open("runs.txt", "w+")     
    res = open("Best_Thresholds.txt", "w+")

    for num_thresholds in range(2, 6):
        for image in os.listdir(os.getcwd() + "\greyscale_count"):
            best_value = 0
            best_thres = []
            for runs in range(1):
                x = GWO(0, 255, num_thresholds, 30, 10, image)
                runfile.writelines("%f " % x.value)
                if x.value > best_value:
                    best_value = x.value
                    best_thres = x.thresholds
            runfile.writelines("\n")
            for each_thres in best_thres:
                res.writelines("%d "%each_thres)
            res.writelines("\n")
        res.writelines("\n\n\n\n")
        runfile.writelines("\n\n\n\n")

    runfile.close()
    res.close()
