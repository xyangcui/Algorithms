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
    check a nan member and regenerate it.
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
    real = data['real']
    obs  = data['obs']
    B    = data['B']
    R    = data['R']
# --------------------------------------------------
# 500 forecast / DA starting indices
# --------------------------------------------------
_, nstep= real.shape
forecast_idx = np.arange(0, nstep, 100)[:tof]
# --------------------------------------------------
# 4 observations over 0.2 TU
# offsets = 2,4,...,20
# --------------------------------------------------
obs_offset = np.arange(1, DA_window + 1, 1)
nobs = len(obs_offset)   # 10
rng = np.random.default_rng(42)
# --------------------------------------------------
# Initial states
#
# member 0     : control
# member 1:nmember+1 : nmember perturbed members
#
# shape = (K, nmember+1, tof)
# --------------------------------------------------
initial_state = np.zeros((K, nmember + 1, tof),dtype=np.float64)
# --------------------------------------------------
# Observations
#
# shape = (K, 4, nmember+1, tof)
# --------------------------------------------------
observation = np.zeros((K, nobs, nmember + 1, tof),dtype=np.float64)

for icase, idx in enumerate(forecast_idx):
    # ==============================================
    # control initial state
    # truth + background error
    # ==============================================
    xb_control = (real[:, idx]+ rng.normal(0.0, bg_std, size=K))
    initial_state[:, 0, icase] = xb_control
    # ==============================================
    # nmember perturbed initial states
    # ==============================================
    perturb = rng.normal(0.0,bg_std[:, None],size=(K, nmember))
    # zero-mean ensemble perturbations
    perturb -= perturb.mean(axis=1, keepdims=True)
    initial_state[:, 1:, icase] = (xb_control[:, None] + perturb)
    # ==============================================
    # original observations for this DA window
    # ==============================================
    y = obs[:, idx + obs_offset]      # (3, 10)
    # control: no extra perturbation
    observation[:, :, 0, icase] = y
    # ==============================================
    # nmember perturbed observations
    # ==============================================
    observation[:, :, 1:, icase] = (y[:, :, None]+ rng.normal(0.0,obs_std,size=(K, nobs, nmember)))

print(initial_state.shape) # (3, 101, 200) (dim,nmember,ncase)
initial_state_bk = initial_state
 
print(observation.shape) # (3, 10, 101, 200) (dim,windows,nmember,ncase)
print(observation[0,0,0,0])
# Ensemble of Data Assimilation
#l96 = lambda x: L96(x,F)
#l96_tlm = lambda x: L96_TLM_operator(x,dt)
#rk4 = lambda x,y: runge_kuta4(x,y,dt)
#DA_module = FourDVar(l96,l96_tlm,rk4)

l96 = lambda x: L96(x,F)
l96_adm = lambda x,y: L96_adm(x,y)
rk4 = lambda x,y: runge_kuta4(x,y,dt)
rk4_adm = lambda x,y,z,q: rk4_nl_adm(x,y,z,q,dt)
DA_module = FourDVar_practical(l96,l96_adm,rk4,rk4_adm)

for ncase in range(tof):
    for nm in range(nmember+1):
        initial_state[:,nm,ncase] = DA_module.four_dims_var_optimizer(initial_state[:,nm,ncase],B,
                                                                      observation[:,:,nm,ncase],R,
                                                                       Dh,h,max_iter=1000,tol=1e-7)
# store EDA
with open('ensembleDA.pkl', 'wb') as f:
    pickle.dump(initial_state, f)
# store true trajectories
x_truth = np.zeros((K,nt+1,tof),dtype=np.float64)
for i in range(tof):
    x_truth[:,:,i] = real[:,i:i+nt+1]
with open('forecast_truth.pkl', 'wb') as f:
    pickle.dump(x_truth, f)    
# store background state
with open('background.pkl', 'wb') as f:
    pickle.dump(initial_state_bk, f)

# DA test
def DA_test(DA):
    xa_start = DA.four_dims_var_optimizer(initial_state[:,0,0],B,observation[:,:,0,0],R,Dh,h,max_iter=1000,tol=1e-5)
    xa = np.zeros((K,nt+1))
    xa[:,0] = xa_start.copy()
    for i in range(nt):
        xa[:,i+1] = runge_kuta4(l96,xa[:,i],dt)

    xb_start = initial_state[:,0,0]
    xb = np.zeros((K,nt+1))
    xb[:,0] = xb_start.copy()
    for i in range(nt):
        xb[:,i+1] = runge_kuta4(l96,xb[:,i],dt)

    x_truth = real[:,forecast_idx[0]:forecast_idx[0]+nt+1]

    RMSEb = np.linalg.norm(xb-x_truth,axis=1)
    RMSEa = np.linalg.norm(xa-x_truth,axis=1)

    print(RMSEb.mean())
    print(RMSEa.mean())

#DA_test(DA_module)
#DA_test(DA_module2)