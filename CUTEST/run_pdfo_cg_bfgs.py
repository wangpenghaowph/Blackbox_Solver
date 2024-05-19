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
import threading
class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout, *args, **kwargs):
    result = [None]
    exception = [None]

    def target_function():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target_function)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"Function {func.__name__} timed out")
        return None
    if exception[0]:
        raise exception[0]
    return result[0]

def pdfo_run(problem_name, ratio=500, timeout=60):
    def target_function():
        problem = pycutest.import_problem(problem_name)
        maxfev = ratio * problem.n
        result = pdfo(problem.obj, problem.x0, method='newuoa', options={'maxfev': maxfev})
        return result
    return run_with_timeout(target_function, timeout)

def cg_scipy_run(problem_name, ratio=500, timeout=60):
    def target_function():
        problem = pycutest.import_problem(problem_name)
        print(f'dimension is {problem.n}, ratio is {ratio}')
        maxfev = ratio * problem.n
        result = minimize(problem.obj, problem.x0, method='CG', jac="2-point", options={'maxiter': maxfev, 'return_all': True})
        return result
    return run_with_timeout(target_function, timeout)

def bfgs_scipy_run(problem_name, ratio=500, timeout=60):
    def target_function():
        problem = pycutest.import_problem(problem_name)
        maxfev = ratio * problem.n
        result = minimize(problem.obj, problem.x0, method='BFGS', jac="2-point", options={'maxiter': maxfev, 'return_all': True})
        return result
    return run_with_timeout(target_function, timeout)
# 查找未测试的问题
def find_untested_problems(all_problems, tested_results):
    tested_problems = {result['Problem'] for result in tested_results}
    untested_problems = [prob for prob in all_problems if prob not in tested_problems]
    print(f'total number of problems: {len(all_problems)}, number of untested problems: {len(untested_problems)}')
    return untested_problems

# 主测试逻辑
def run_tests_for_untested(untested_problems, run_function, direct_res, key_res, save_path_direct, save_path_key, ratio=500, timeout=3600):
    for prob in tqdm.tqdm(untested_problems):
        print(f"Using {run_function.__name__}")
        result = run_function(prob, ratio, timeout)
        if result is not None:
            direct_res.append(result)
            if run_function.__name__ == 'pdfo_run':
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
        else:
            print(f"Problem {prob} failed or timed out with {run_function.__name__}")
            direct_res.append("timeout")
            key_res.append({
                'Problem': prob,
                'Objective': "timeout",
                'Success': "timeout",
                'solution': "timeout",
                'Message': "timeout",
                'status': "timeout",
                'nfev': "timeout",
                'niter': "timeout" if run_function.__name__ != 'pdfo_run' else "timeout",
                'fhist': "timeout" if run_function.__name__ == 'pdfo_run' else None
            })

        save_data_with_pickle(direct_res, save_path_direct)
        save_data_with_pickle(key_res, save_path_key)
        if result is not None:
            print(f"Problem {prob} is solved with status {result.status} and success={result.success}")
        else:
            print(f"Problem {prob} timed out and is recorded with 'timeout' values.")


# 数据加载和保存函数
def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_from_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        print(f"File {filename} not found")
        return []  # 如果文件不存在，返回一个空列表