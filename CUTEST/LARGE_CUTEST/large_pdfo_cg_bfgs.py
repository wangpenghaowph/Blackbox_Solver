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
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
print(f'current dir={os.getcwd()}')
import run_pdfo_cg_bfgs
from run_pdfo_cg_bfgs import *
import importlib
importlib.reload(run_pdfo_cg_bfgs)
# 加载问题列表和结果
df_problems = pd.read_csv("Loader_CUTEST/unconstrained_problems_Large.csv")
all_problems = df_problems["data"].tolist()

pdfo_direct_path = 'pkl_file/large_pdfo_direct.pkl'
pdfo_key_path = 'pkl_file/large_pdfo_key.pkl'
cg_direct_path = 'pkl_file/large_cg_direct.pkl'
cg_key_path = 'pkl_file/large_cg_key.pkl'
bfgs_direct_path = 'pkl_file/large_bfgs_direct.pkl'
bfgs_key_path = 'pkl_file/large_bfgs_key.pkl'
pdfo_direct_res = load_data_from_pickle(pdfo_direct_path)
pdfo_key_res = load_data_from_pickle(pdfo_key_path)
cg_direct_res = load_data_from_pickle(cg_direct_path)
cg_key_res = load_data_from_pickle(cg_key_path)
bfgs_direct_res = load_data_from_pickle(bfgs_direct_path)
bfgs_key_res = load_data_from_pickle(bfgs_key_path)

# 获取未测试的问题
untested_pdfo = find_untested_problems(all_problems, pdfo_key_res)
untested_cg = find_untested_problems(all_problems, cg_key_res)
untested_bfgs = find_untested_problems(all_problems, bfgs_key_res)

run_tests_for_untested(untested_pdfo, pdfo_run, pdfo_direct_res, pdfo_key_res, pdfo_direct_path, pdfo_key_path, ratio=500, timeout = 1800)
run_tests_for_untested(untested_cg, cg_scipy_run, cg_direct_res, cg_key_res, cg_direct_path, cg_key_path, ratio=500, timeout = 1800)
run_tests_for_untested(untested_bfgs, bfgs_scipy_run, bfgs_direct_res, bfgs_key_res, bfgs_direct_path, bfgs_key_path, ratio=500, timeout = 1800)