import numpy as np

# maximum likelihood for a set of datapoints is calculated by multiplying
# probability of each datapoint, more the likelihood better the model classifiying them

# Cross Entropy is calculated by multiplying actual class of datapoint with negative log 
# probability of dataset being in that class, less the better

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def binaryClass_cross_entropy(Y,P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

def multiClass_cross_entropy(Y,P,classes):
    pass
    
