
# This module is to construct local model in each iteration
"""
Input:
A list of vectors
"""

"""
Call Module(function): 
Sample.py
"""

import Sample

def Construct_Model(x, directions, history, iter):
    # TODO: Define how the model is constructed
    if method_construct_model == 'linear':
        c = Linear_Model(directions, trust_radius, nfev_counter)
    if method_construct_model == 'quadratic':
        Q = Quadratic_Model(directions, trust_radius, nfev_counter)
    # define a local model function
    def local_model(x):
        pass 
    return local_model

def Linear_Model(nfev, directions, trust_radius):
    #n = len(directions)
    #model = []
    #for i in range(n):
    #    model.append([directions[i], 1])
    c=[]
    pass

def Quadratic_Model(nfev, directions, trust_radius):
    Q=[]
    c=[]
    return Q, c
