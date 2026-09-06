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
bg_coeff = 0.5 # use bg_coeff*B to perturb true value
tof  = 200   # time of forecast
nmember = 50   # number of members
obs_coeff = 0.3
DA_dt     = 0.005   # dt of DA. propagate slowly to include small dynamics
DA_window = 0.2 # length of DA window 0.2tu.
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
B_perturb = bg_coeff*B
# --------------------------------------------------
# Observations
# shape = (K, nobs, nmember+1, tof)
# --------------------------------------------------
observation = np.zeros((K, nobs, nmember + 1, tof),dtype=np.float64)
R = np.diag(obs_coeff * np.diag(B))

for icase, idx in enumerate(forecast_idx):
    # ==============================================
    # control initial state
    # truth + background error
    # ==============================================
    perturb = rng.multivariate_normal(mean=np.zeros(K),cov=B_perturb)
    xb_control = real[:, idx] + perturb
    initial_state[:, :, icase] = xb_control[:,None]
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
    obs_error = rng.multivariate_normal(mean=np.zeros(K),cov=R,size=nobs)
    y = x_truth[:,1::2] + obs_error.T
    observation[:, :, 0, icase] = y
    # ==============================================
    # nmember perturbed observations
    # ==============================================
    obs_perturb = rng.multivariate_normal(mean=np.zeros(K),cov=R,size=(nmember,nobs)).T
    observation[:, :, 1:, icase] = y[:, :, None] + obs_perturb

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
# DA test
def DA_test(DA):
    nm = 1; nc = 20
    xa_start = DA.four_dims_var_optimizer(initial_state[:,nm,nc],B_perturb,observation[:,:,nm,nc],R,obs_idx,n,Dh,h,max_iter=1000,tol=1e-7)
    xa = np.zeros((K,nt+1))
    xa[:,0] = xa_start.copy()
    for i in range(nt):
        xa[:,i+1] = runge_kuta4(l96,xa[:,i],dt)

    xb_start = initial_state[:,nm,nc]
    xb = np.zeros((K,nt+1))
    xb[:,0] = xb_start.copy()
    for i in range(nt):
        xb[:,i+1] = runge_kuta4(l96,xb[:,i],dt)

    x_truth = real[:,forecast_idx[0]:forecast_idx[0]+nt+1]

    RMSEb = np.linalg.norm(xb-x_truth,axis=1)
    RMSEa = np.linalg.norm(xa-x_truth,axis=1)

    print(f"background: {RMSEb.mean()}")
    print(f"analysis: {RMSEa.mean()}")


def taylor_test_4dvar(fourdvar,x,xb,B,y,R,idx,H,h,N,seed=42):
    """
    Taylor test for FourDVar_practical gradient.

    Check:
        J(x + eps*d)
        = J(x) + eps * gradJ(x)^T d + O(eps^2)

    If gradient is correct:
        first-order residual ~ O(eps^2)
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Random normalized perturbation direction
    # --------------------------------------------------
    d = rng.normal(size=x.shape)
    d /= np.linalg.norm(d)

    # --------------------------------------------------
    # R inverse
    # Important: use full R if R is not diagonal
    # --------------------------------------------------
    R_inv = np.linalg.inv(R)

    # --------------------------------------------------
    # Base cost and gradient
    # --------------------------------------------------
    J0 = fourdvar.cost_function(x,xb,B,R_inv,idx,y,h,N)

    grad = fourdvar.gradient(x,xb,B,y,R_inv,idx,H,h,N)

    directional_grad = np.dot(grad, d)

    print("J(x) =", J0)
    print("||grad|| =", np.linalg.norm(grad))
    print("grad^T d =", directional_grad)
    print()

    print(
        f"{'eps':>12s} "
        f"{'|J(x+ed)-J(x)|':>20s} "
        f"{'Taylor residual':>20s} "
        f"{'ratio':>12s}"
    )

    previous = None

    eps_list = 10.0 ** (-np.arange(1, 9))

    residuals = []

    for eps in eps_list:

        x_eps = x + eps * d

        J_eps = fourdvar.cost_function(x_eps,xb,B,R_inv,idx,y,h,N)
        # zeroth-order difference
        diff0 = abs(J_eps - J0)

        # first-order Taylor residual
        residual = abs(J_eps- J0- eps * directional_grad)

        residuals.append(residual)

        if previous is None:
            ratio = np.nan
        else:
            ratio = previous / residual

        print(
            f"{eps:12.1e} "
            f"{diff0:20.8e} "
            f"{residual:20.8e} "
            f"{ratio:12.4f}"
        )

        previous = residual

    return eps_list, np.array(residuals)

#x_test = initial_state[:,0,0].copy()
#eps_list, residuals = taylor_test_4dvar(fourdvar=DA_module,x=x_test,xb=initial_state[:,0,0],B=B,y=y,R=R,idx=obs_idx,H=Dh,h=h,N=n)

DA_test(DA_module)