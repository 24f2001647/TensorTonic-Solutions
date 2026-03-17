import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y=np.array(y) #if array not there, have to return 0
    values,counts=np.unique(y,return_counts=True)
    prob = counts/counts.sum() #ensure log is stable
    entropy = -np.sum(prob*np.log2(prob))
    return entropy