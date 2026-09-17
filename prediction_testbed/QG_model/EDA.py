import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from baro_utilis import BARO_VORT, ift, ft
from DA_utilis import FourDVar_practical
from datetime import datetime
import numpy as np
import pickle, json, time, os

## set model parameters
with open("config.json", "r") as f:
    config = json.load(f)
## model configration
dt = config['model']['dt']
nx = config['model']['nx']
ny = config['model']['ny']
nobs = config['observations']['nobs']
# cases of forecast
nfcst = config['forecast_evaluation']['n_cases']
# number of perturbed members
eda_members = config['eda']['default_members']

model = BARO_VORT(config['model'])

## DA configration
da_tu = config['data_assimilation']['cycle_interval_tu']
da_window = config['data_assimilation']['four_d_var_window_tu']
da_steps = config['data_assimilation']['four_d_var_window_steps']
obs_steps = int(da_window/da_tu)
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


HC = np.zeros((nobs, nx * ny))
for i, idx in enumerate(obs_idx):
    HC[i, idx] = 1.0

def H(x):
    return HC

def h(x):
    return x[obs_idx]

##
def propagator(z):
    global nx, ny, forcet, ft
    z_spec = ft(z.reshape(nx, ny))
    z_next_spec = model.bve_propagator(z_spec, forcet)
    return ift(z_next_spec).reshape(-1)

def adm_propagator(z_state,z_adj):
    global nx, ny, forcet, ft
    state_spec = ft(z_state.reshape(nx, ny))
    adj_spec   = ft(z_adj.reshape(nx, ny))

    adj_prev_spec = model.bve_adm_propagator(model,state_spec,adj_spec,forcet)

    return ift(adj_prev_spec).reshape(-1)

z_initial_control_t = z_initial_control.reshape(nfcst,nx*ny)
z_initial_pert_t = z_initial_pert.reshape(nfcst,eda_members,nx*ny)
obs_control = obs.reshape(nfcst,len(obs_step_idx),-1)
obs_perturb = obs_pert.reshape(nfcst,eda_members,len(obs_step_idx),-1)

DA_module = FourDVar_practical(model_propagator=propagator,
                               model_ADM_propagator=adm_propagator)

z_analysis = np.zeros((eda_members+1,nx*ny),dtype=np.float32)

t0 = time.perf_counter()
for icase in range(66,69):
    z_analysis = np.zeros((eda_members+1,nx*ny),dtype=np.float32)
    # control forecast
    print(f'case {icase+1} begin. {datetime.now().strftime("%m/%d %H:%M:%S")}')
    print(f' control forecast.')
    z_initial = z_initial_control_t[icase]
    B = B_init
    lam = lamB_init
    observation = obs_control[icase]
    R = obs_std_control**2
    z_analysis[0,:] = DA_module.four_dims_var_optimizer_scipy(xb=z_initial,
                                                     B=B,
                                                     y=observation.T,
                                                     R=R,
                                                     idx=obs_step_idx,
                                                     n=da_steps,
                                                     H=H,
                                                     h=h,
                                                     lam=lam, 
                                                    )
    print(f'perturbed forecast begin. {datetime.now().strftime("%m/%d %H:%M:%S")}')
    for imember in range(eda_members):
        print(f'member {imember+1}.')
        z_initial = z_initial_pert_t[icase,imember]
        B = B_pert
        lam = lamB_pert
        observation = obs_perturb[icase,imember]
        R = obs_std_pert**2        
        z_analysis[imember+1,:] = DA_module.four_dims_var_optimizer_scipy(xb=z_initial,
                                                     B=B,
                                                     y=observation.T,
                                                     R=R,
                                                     idx=obs_step_idx,
                                                     n=da_steps,
                                                     H=H,
                                                     h=h,
                                                     lam=lam, 
                                                    )
    # store
    z_analysis_regrid = z_analysis.reshape(1+eda_members,nx,ny)
    with open(f'./eda_members/eda_{icase + 1:03d}.pkl', 'wb') as f:
        pickle.dump(z_analysis_regrid, f)

t1 = time.perf_counter()
print(f"consumption: {t1 - t0:.4f} s")

