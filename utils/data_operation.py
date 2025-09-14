import numpy as np 
import math

def euclidean_distance(x1,x2):
    dist = 0
    for i in range(len(x1)):
        dist += pow((x1[i] - x2[i]),2)
    return math.sqrt(dist)

def accuracy_score(y_true,y_pred):
    acc = np.sum(y_true == y_pred,axis=0)/len(y_pred)
    return acc