import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm
from solve_ode import runge_kuta4, rk4_nl_adm
#from DA_utilis import h,Dh, FourDVar_practical
from DA_utilis_fixed import h, Dh, FourDVar_practical
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
bg_coeff =   1.5# use bg_coeff*B to perturb true value
bg_control = 0.8
tof  = 200   # time of forecast
nmember = 50   # number of members
obs_control = 0.1
obs_coeff   = 0.15
DA_dt     = 0.005   # dt of DA. propagate slowly to include small dynamics
DA_window = 0.15 # length of DA window 0.2tu.
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
nobs = int(DA_window/DA_dt)//2   # 10
obs_idx = np.arange(1,nobs*2,2)
rng = np.random.default_rng(42)
# --------------------------------------------------
# Initial states
# member 0     : control
# member 1:nmember+1 : nmember perturbed members
# shape = (K, nmember+1, tof)
# --------------------------------------------------
initial_state = np.zeros((K, nmember + 1, tof),dtype=np.float64)
B_control = bg_control*B
B_perturb = bg_coeff*B
# --------------------------------------------------
# Observations
# shape = (K, nobs, nmember+1, tof)
# --------------------------------------------------
observation = np.zeros((K, nobs, nmember + 1, tof),dtype=np.float64)
R_control = obs_control*R
R_perturb = obs_coeff*R

for icase, idx in enumerate(forecast_idx):
    # ==============================================
    # control initial state
    # truth + background error
    # ==============================================
    xb_error = rng.multivariate_normal(mean=np.zeros(K),cov=B_control)
    xb_control = real[:, idx] + xb_error
    initial_state[:, 0, icase] = xb_control
    ## perturb it
    perturb = rng.multivariate_normal(mean=np.zeros(K),cov=B_perturb,size=(nmember)).T
    initial_state[:, 1:, icase] = real[:, idx][:,None] + perturb  
    # ==================================================
    # 3. Generate pseudo observations
    # Start from truth at background time
    # then integrate model by DA_dt each time
    # ==================================================
    x_truth = np.zeros((K,2*nobs))
    x_truth[:,0] = real[:, idx].copy()
    for iobs in range(nobs*2-1):
        x_truth[:,iobs+1] = runge_kuta4(lambda x: L96(x,F),x_truth[:,iobs],DA_dt)
    
    ## psedo observation
    obs_error = rng.multivariate_normal(mean=np.zeros(K),cov=R_control,size=nobs)
    y = x_truth[:,1::2] + obs_error.T
    observation[:, :, 0, icase] = y
    # ==============================================
    # nmember perturbed observations
    # ==============================================
    obs_perturb = rng.multivariate_normal(mean=np.zeros(K),cov=R_perturb,size=(nmember,nobs)).T
    observation[:, :, 1:, icase] = x_truth[:,1::2][:, :, None] + obs_perturb

print(initial_state.shape) # (3, 101, 200) (dim,nmember,ncase)
print(observation.shape) # (3, 10, 101, 200) (dim,windows,nmember,ncase)
print(obs_idx)
initial_state_bk = initial_state

# Ensemble of Data Assimilation
l96 = lambda x: L96(x,F)
l96_adm = lambda x,y: L96_adm(x,y)
rk4 = lambda x,y: runge_kuta4(x,y,DA_dt)
rk4_adm = lambda x,y,z,q: rk4_nl_adm(x,y,z,q,DA_dt)
DA_module = FourDVar_practical(l96,l96_adm,rk4,rk4_adm)
n = int(DA_window/DA_dt)
def start_EDA():

    for ncase in range(tof):
        for nm in range(nmember+1):
            if nm == 0:
                Bda = B_control; Rda = R_control
            else:
                Bda = B_perturb; Rda = R_perturb

            initial_state[:,nm,ncase] = DA_module.four_dims_var_optimizer_scipy(initial_state[:,nm,ncase],Bda,
                                                                      observation[:,:,nm,ncase],Rda,obs_idx,n,
                                                                       Dh,h,max_iter=1000,tol=1e-5,verbose=False)
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

start_EDA()