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
nobs = config['observations']['nobs']

model = BARO_VORT(config['model'])

## DA configration
da_tu = config['data_assimilation']['cycle_interval_tu']
da_window = config['data_assimilation']['four_d_var_window_tu']


with open('climate_simulation.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    forcet = data_dict['forcing']['data']

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
# location of observation
obs_idx = data_dict['obs_idx']['data']
# step of observation
obs_step_idx = data_dict['obs_step_idx']['data']

for name, x in {
    "z_initial_control": z_initial_control,
    "B_init": B_init,
    "z_initial_pert": z_initial_pert,
    "B_pert": B_pert,
    "obs": obs,
    "obs_std_control": obs_std_control,
    "obs_pert": obs_pert,
    "obs_std_pert": obs_std_pert,
    "obs_idx": obs_idx,
}.items():
    print(f"{name}.shape = {x.shape}")

def H(x):
    global obs_idx, nobs
    H = np.zeros((nobs, nx * ny))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0

    return H

def h(x):
    global obs_idx
    xobs = np.zeros(nobs)
    for i, idx in enumerate(obs_idx):
        xobs[i] = x[idx]

    return xobs

##
def propagator(z):
    global nx, ny, forcet, ft
    zt = z.reshape(nx,ny)
    zt_spec = ft(zt)
    zt = model.bve_propagator(zt_spec,forcet)
    z  = ift(zt)

    return z.ravel()

def adm_propagator(z_state,z_adj):
    global nx, ny, forcet, ft
    # nonliear state
    z_state_temp = z_state.reshape(nx,ny)
    z_state_spec = ft(z_state_temp)
    # adjoint state
    z_adj_temp = z_adj.reshape(nx,ny)
    z_adj_spec = ft(z_adj_temp)
    # integration
    zt = model.bve_adm_propagator(model,z_state_spec,z_adj_spec,forcet)
    z  = ift(zt)

    return z.ravel()

z_initial_control_t = z_initial_control.reshape(100,nx*ny)

DA_module = FourDVar_practical(model_propagator=propagator,
                               model_ADM_propagator=adm_propagator)

z_initial = z_initial_control_t[0]
B = B_init
lam = lamB_init
observation = obs.reshape(100,5,-1)[0]
R = obs_std_control

z_analysis = DA_module.four_dims_var_optimizer_scipy(xb=z_initial,
                                                     B=B_init,
                                                     y=observation.T,
                                                     R=R,
                                                     idx=obs_step_idx,
                                                     n=int(da_window/dt),
                                                     H=H,
                                                     h=h,
                                                     lam=lam,
                                                     )