import numpy as np

# Acts as an activation function for multiclass learning problems

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp = np.exp(L)
    sum = np.sum(exp)
    results = []
    for i in exp:
        results.append(i/sum)

    return results