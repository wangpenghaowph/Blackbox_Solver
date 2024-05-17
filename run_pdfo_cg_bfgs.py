import os
import pandas as pd
import pickle
import tqdm
import scipy
from scipy.optimize import minimize
import pycutest
from pdfo import pdfo
import sys
import numpy as np
def pdfo_run(problem_name, ratio = 500):
    problem = pycutest.import_problem(problem_name)
    maxfev = ratio * problem.n
    result = pdfo(problem.obj, problem.x0, method='newuoa', options={'maxfev': maxfev})
    return result

def cg_scipy_run(problem_name, ratio = 500):
    problem = pycutest.import_problem(problem_name)
    maxfev = ratio * problem.n
    result = minimize(problem.obj, problem.x0, method='CG', jac="2-point", options={'maxiter': maxfev, 'return_all': True})
    return result

def bfgs_scipy_run(problem_name, ratio = 500):
    problem = pycutest.import_problem(problem_name)
    maxfev = ratio * problem.n
    result = minimize(problem.obj, problem.x0, method='BFGS', jac="2-point", options={'maxiter': maxfev, 'return_all': True})
    return result

# 查找未测试的问题
def find_untested_problems(all_problems, tested_results):
    tested_problems = {result['Problem'] for result in tested_results}
    untested_problems = [prob for prob in all_problems if prob not in tested_problems]
    return untested_problems

# 主测试逻辑
def run_tests_for_untested(untested_problems, run_function, direct_res, key_res, save_path_direct, save_path_key, ratio=500):
    for prob in tqdm.tqdm(untested_problems):
        if run_function.__name__ == 'pdfo_run':
            print("Using PDFO")
            result = run_function(prob, ratio)
            direct_res.append(result)
            key_res.append({
                'Problem': prob,
                'Objective': result.fun,
                'Success': result.success,
                'solution': result.x,
                'Message': result.message,
                'status': result.status,
                'nfev': result.nfev,
                'fhist': result.fun_history
            })
        else:
            print(f"Using {run_function.__name__}")
            result = run_function(prob, ratio)
            direct_res.append(result)
            key_res.append({
                'Problem': prob,
                'Objective': result.fun,
                'Success': result.success,
                'solution': result.x,
                'Message': result.message,
                'status': result.status,
                'nfev': result.nfev,
                'niter': result.nit
            })
        save_data_with_pickle(direct_res, save_path_direct)
        save_data_with_pickle(key_res, save_path_key)
        print(f"Problem {prob} is solved with status {result.status} and success={result.success}")

# 数据加载和保存函数
def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_from_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return []  # 如果文件不存在，返回一个空列表