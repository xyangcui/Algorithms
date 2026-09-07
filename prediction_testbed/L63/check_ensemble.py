import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm
from solve_ode import runge_kuta4, rk4_nl_adm
from DA_utilis import h,Dh, FourDVar_practical
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
# load background initial state
with open('background.pkl', 'rb') as f:
    initial_state_bk = pickle.load(f)
# load forecast ensemble [dim,ensemble,time]
with open('ensembleDA.pkl', 'rb') as f:
    initial_state = pickle.load(f)
# load forecast truth
with open('forecast_truth.pkl', 'rb') as f:
    truth = pickle.load(f)

_,nmember,ncase = initial_state.shape
# organize observation
nobs = int(DA_window/DA_dt)//2   # 10
obs_idx = np.arange(1,nobs*2,2)
rng = np.random.default_rng(42)
R_perturb = obs_coeff*R
B_perturb = bg_coeff*B
# Ensemble of Data Assimilation
l96 = lambda x: L96(x,F)
l96_adm = lambda x,y: L96_adm(x,y)
rk4 = lambda x,y: runge_kuta4(x,y,DA_dt)
rk4_adm = lambda x,y,z,q: rk4_nl_adm(x,y,z,q,DA_dt)
DA_module = FourDVar_practical(l96,l96_adm,rk4,rk4_adm)
n = int(DA_window/DA_dt)

# check each case and each member
if initial_state.any() == np.nan:
    for icase in range(ncase):
        for inumber in range(nmember):
            state = initial_state[:,inumber,icase]
            real  = truth[:,0,icase]
            if state.any() == np.nan:
                print(f"number {inumber+1} of case {icase+1} needs regenerated.")
                # perturb observation again to generate it.
                x_truth = np.zeros((K,2*nobs))
                x_truth[:,0] = real.copy()
                for iobs in range(nobs*2-1):
                    x_truth[:,iobs+1] = runge_kuta4(lambda x: L96(x,F),x_truth[:,iobs],DA_dt)
                ## psedo observation
                obs_error = rng.multivariate_normal(mean=np.zeros(K),cov=R_perturb,size=nobs)
                # DA
                initial_state[:,inumber,icase] = DA_module.four_dims_var_optimizer_scipy(xb=initial_state[:,inumber,icase],
                                                                                B=B_perturb,
                                                                                y=x_truth+obs_error.T,
                                                                                R=R_perturb,
                                                                                idx=obs_idx,
                                                                                n=n,
                                                                                H=Dh,
                                                                                h=h,
                                                                                max_iter=1000,
                                                                                tol=1e-5,
                                                                                verbose=True,
                                                                                return_history=False)
    with open('ensembleDA.pkl', 'wb') as f:
        pickle.dump(initial_state, f) 
else:
    print(f"All initial states are correct.")
    ## integrate to the end of an assimilation window.
    for icase in range(ncase):
        for inumber in range(nmember):
            for i in range(n):
                initial_state[:,inumber,icase] = runge_kuta4(lambda x: L96(x,F),initial_state[:,inumber,icase],DA_dt)
    with open('ensembleDA_end_of_DAwindow.pkl', 'wb') as f:
        pickle.dump(initial_state, f)  