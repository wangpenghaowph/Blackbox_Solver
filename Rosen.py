import numpy as np

def rosenbrock(x):
    '''
    input:
    x -- ndarray; current point
    '''
    a = 1.0
    b = 100.0
    return sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)