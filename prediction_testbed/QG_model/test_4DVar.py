import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from baro_utilis import BARO_VORT, ift, ft
from DA_utilis import FourDVar_practical
import numpy as np
import pickle, json

## set model parameters
with open("config.json", "r") as f:
    config = json.load(f)
## model configration
dt = config['model']['dt']
nx = config['model']['nx']
ny = config['model']['ny']

model = BARO_VORT(config['model'])

## DA configration
da_tu = config['data_assimilation']['cycle_interval_tu']
da_window = config['data_assimilation']['four_d_var_window_tu']

## DA data
with open('da_processing.pkl', 'rb') as f:
    data_dict = pickle.load(f)
# initial state of control forecast at the start of assimilation window
z_initial_control = data_dict['z_initial_control']['data']
# factors of control forecast
B_init = data_dict['B_init']['data']
# lambda to regularize B_init
lamB_init = data_dict['lam_B_init']['data']
# initial state of perturbed forecast at the start of assimilation window
z_initial_pert = data_dict['z_initial_pert']['data']
# factors of control forecast perturbed forecasts
B_pert = data_dict['B_pert_total']['data']
# lambda to regularize B_pert
lamB_pert = data_dict['lam_B_pert']['data']
# observation of control forecast
obs = data_dict['obs']['data']
# representative error covariance vector, to disturb it.
obs_std_control = data_dict['obs_control_std']['data']
# observation of perturbed forecast
obs_pert = data_dict['obs_pert']['data']
# representative error covariance vector, to disturb it.
obs_std_pert = data_dict['obs_pert_total_std']['data']


