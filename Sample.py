# This module is to sample a given number of points
"""
Input:
A list of vectors
"""

"""
Call Module(function): 
Counter.py (nfev_counter, ngrad_counter)
Sample.py
"""
import History_Manager

def evaluate(obj_fun, x, iter, stage, history):
    obj = obj_fun(x)
    history.add_history(x, obj, iter, stage)
    return obj

def sample(obj_fun, x, method_direct_search, method, iter, stage, history):
    results = []
    if method_direct_search == 'orthogonal' and stage == 'DS':
        # TODO:points = 
        pass
    else:
        # TODO:points = 
        pass
    for x in points:
        result = evaluate(obj_fun, x, stage, iter, history)
        results.append(result)
    return results