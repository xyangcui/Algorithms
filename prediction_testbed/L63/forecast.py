import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96
from solve_ode import runge_kuta4
import numpy as np
import pickle


# load background initial state
with open('background.pkl', "rb") as f:
    BK_start = pickle.load(f)
# load EDA initial state
with open('ensembleDA_end_of_DAwindow.pkl', "rb") as f:
    EDA_start = pickle.load(f)
# load singular vectors
with open('singular_vectors.pkl', "rb") as f:
    SV_start = pickle.load(f)
# load truth
with open('forecast_truth.pkl', "rb") as f:
    xt = pickle.load(f)

## combine EDA and singular vectors
# 1. obtain EDA anomaly
EDA_mean = np.mean(EDA_start,axis=1,keepdims=False)
EDA_anom = EDA_start[:,1:,:] - EDA_mean[:,None,:]
# Analysis (EDA control) +- EDA perturbation +- SV perturbation
K,nmember,ncase = EDA_start.shape
fcst_start_control = EDA_start[:,0,:][:,None,:]
fcst_start_pos = EDA_start[:,0,:][:,None,:]+EDA_anom+SV_start.reshape(K,nmember-1,ncase)
fcst_start_neg = EDA_start[:,0,:][:,None,:]-EDA_anom-SV_start.reshape(K,nmember-1,ncase)
fcst_start = np.concatenate([fcst_start_control,fcst_start_pos,fcst_start_neg],axis=1)

