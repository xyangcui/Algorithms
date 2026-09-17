import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from baro_utilis import BARO_VORT, ift, ft
from fcst_utils import singular_vectors
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

da_steps = config['data_assimilation']['four_d_var_window_steps']

sv_tu = config['singular_vectors']['optimization_interval_tu']
sv_n  = config['singular_vectors']['leading_vectors']

with open('climate_simulation.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    forcet = data_dict['forcing']['data']


# projection function, currently set to identified matrix.
def P(x):
    return x
# a function to calculate Euclidian distance.
def euclidian_norm(K):
    '''normalize x by its distance.'''
    return np.full(K,1,dtype=float) 
# total energy norm.

# TLM
def tlm_propagator(dz,z_spec,N):
    dz_spec = ft(dz.reshape(nx, ny))
    for i in range(N):
        z_spec, dz_spec = model.bve_tlm_propagator(model, z_spec, dz_spec, forcet)

    return ift(dz_spec).reshape(-1)
# ADM
def adm_propagator(z_adj,zbase,N):
    z_adj_spec = ft(z_adj.reshape(nx, ny))
    # step2: backward integrating
    for i in range(N, 0, -1):
        z_spec = zbase[i-1,:]
        z_adj_spec = model.bve_adm_propagator(model,z_spec,z_adj_spec,forcet)

    return ift(z_adj_spec).reshape(-1)


if os.path.exists('./eda_members/eda.pkl'):
    with open('./eda_members/eda.pkl', 'rb') as f:
        da = pickle.load(f)
else:
    initial_state = np.zeros((nfcst,1+eda_members,nx,ny),dtype=np.complex128)
    for icase in range(nfcst):
        with open(f'./eda_members/eda_{icase+1:03d}.pkl', 'rb') as f:
            da = pickle.load(f)
        for im in range(1+eda_members):
            initial_state[icase,im] = ft(da[im])
            for i in range(da_steps):
                initial_state[icase,im] = model.bve_propagator(initial_state[icase,im], forcet)
    # store
    da = ift(initial_state)
    with open('./eda_members/eda.pkl', 'wb') as f:
        pickle.dump(da,f)

t0 = time.perf_counter()
for icase in range(nfcst):
    print(f'case {icase+1} begin. {datetime.now().strftime("%m/%d %H:%M:%S")}')
    da_control = da[icase,0,:,:]
    control_state = ft(da_control)
    control_states = np.zeros((1+int(sv_tu/dt),nx,ny),dtype=np.complex128)
    control_states[0] = control_state
    for i in range(int(sv_tu/dt)):
        control_states[i+1] = model.bve_propagator(control_states[i], forcet)

    da_perturb = da[icase,1:,:,:].reshape(eda_members,nx*ny)
    da_pert_anom  = da_perturb - da_perturb.mean(axis=0, keepdims=True)

    sv = singular_vectors(m=nx*ny, 
                      nsv=sv_n, 
                      scale=3, 
                      tol=1e-6, 
                      P=P, 
                      r0=euclidian_norm(nx*ny),
                      rf=euclidian_norm(nx*ny),
                      TLM = lambda x: tlm_propagator(x,control_state,int(sv_tu/dt)),
                      ADM = lambda x: adm_propagator(x,control_states,int(sv_tu/dt)),
                      nmember=eda_members,
                      Da=da_pert_anom.T,
                      rescale=0.45)
    # store it
    sv_regrid = sv.reshape(eda_members,nx,ny)
    with open(f'./sv_members/sv_{icase+1:03d}.pkl', 'wb') as f:
        pickle.dump(sv_regrid, f)

t1 = time.perf_counter()
print(f"consumption: {t1 - t0:.4f} s")