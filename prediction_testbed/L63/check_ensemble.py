import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_TLM_operator, L96_adm, L96_tlm
from solve_ode import runge_kuta4, rk4_nl_adm, rk4_nl_tlm
from DA_utilis import FourDVar,h,Dh, FourDVar_practical
import numpy as np
import pickle

'''
    Ensemble of Data Assimilation (EDA)
    The procedure to generate a forecast ensemble by perturbing model and observation.
    Firstly, a control number is generated without any perturbation.
    Then, perturb both to generate a hierachy of models and observations.
'''
dt   = 0.05  # time unit
K    = 40
F    = 8.
tmax = 3     # max integration
nt   = int(tmax/dt) 
bg_std = np.full(K,2.)  # perturb to generate initial state
tof  = 200   # time of forecast
nmember = 50   # number of members
obs_std = 1.5   # std of perturbed observation
DA_window = int(0.4/dt) # length of DA window 0.4tu.
# load ture value and observation
with open('database.pkl', 'rb') as f:
    data = pickle.load(f)
    obs  = data['obs']
    B    = data['B']
    R    = data['R']
# load background initial state
with open('background.pkl', 'rb') as f:
    initial_state_bk = pickle.load(f)
# load forecast ensemble [dim,ensemble,time]
with open('ensembleDA.pkl', 'rb') as f:
    initial_state = pickle.load(f)

_,nmember,ncase = initial_state.shape
# organize observation
rng = np.random.default_rng(42)
_, nstep= obs.shape
forecast_idx = np.arange(0, nstep, 100)[:tof]
obs_offset = np.arange(1, DA_window + 1, 1)
nobs = len(obs_offset)   # 10
observation = np.zeros((K, nobs, nmember + 1, tof),dtype=np.float64)
for icase, idx in enumerate(forecast_idx):
    y = obs[:, idx + obs_offset]      
    observation[:, :, 0, icase] = y
    observation[:, :, 1:, icase] = y[:, :, None]

# Ensemble of Data Assimilation
l96 = lambda x: L96(x,F)
l96_adm = lambda x,y: L96_adm(x,y)
rk4 = lambda x,y: runge_kuta4(x,y,dt)
rk4_adm = lambda x,y,z,q: rk4_nl_adm(x,y,z,q,dt)
DA_module = FourDVar_practical(l96,l96_adm,rk4,rk4_adm)

# check each case and each member
if initial_state.any() == np.nan:
    for icase in range(ncase):
        for inumber in range(nmember):
            state = initial_state[:,inumber,icase]
            if state.any() == np.nan:
                print(f"number {inumber+1} of case {icase+1} needs regenerated.")
                # perturb observation again to generate it.
                obs = observation[:,:,inumber,icase] + rng.normal(0.0,obs_std,size=(K, nobs))
                # DA
                initial_state[:,inumber,icase] = DA_module.four_dims_var_optimizer(initial_state_bk[:,inumber,icase],B,
                                                                               obs,R,
                                                                               Dh,h,max_iter=1000,tol=1e-7)
    with open('ensembleDA.pkl', 'wb') as f:
        pickle.dump(initial_state, f) 
else:
    print(f"All initial states are correct.")