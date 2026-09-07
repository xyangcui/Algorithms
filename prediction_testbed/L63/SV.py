import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm, L96_tlm
from solve_ode import  runge_kuta4, rk4_nl_adm, rk4_nl_tlm
from fcst_utils import singular_vectors
import numpy as np
import pickle

dt   = 0.05  # time unit
K    = 40
F    = 8.
tmax = 3     # max integration
nt   = int(tmax/dt) 
bg_std = np.full(K,2.)  # perturb to generate initial state
nmember = 50   # number of members
obs_std = 1.5   # std of perturbed observation
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
with open('ensembleDA_end_of_DAwindow.pkl', 'rb') as f:
    initial_state = pickle.load(f)
_,nmember,ncase = initial_state.shape
# estimate analysis error vector.
subset = initial_state[:, 1:, :]   # [K, nmember-1, ncase]
Da = subset - subset.mean(axis=1, keepdims=True)
# projection matrix, currently set to identified matrix.
P = np.eye(K)
# a function to calculate total energy metrics.
def total_energy_norm(K):
    '''normalize x by its total energy.'''
    return np.full(K,1,dtype=float) 
# function to integrate TLM. (only needs input as self-variable)
def TLM(x,zt,N,dt):
    for i in range(N):
        zt, x = rk4_nl_tlm(lambda x,y: L96_tlm(x,y),lambda x: L96(x,F),zt,x,dt)
    return x
# function to integrate ADM. (only needs input as self-variable)
def ADM(x,zt,N,dt):
    global K
    # step1: get trajectory
    zbase = np.zeros((K,N+1),dtype=np.float64)
    zbase[:,0] = zt
    for i in range(N):
        zbase[:,i+1] = runge_kuta4(lambda x: L96(x,F),zbase[:,i],dt)
    # step2: backward integrating
    for i in range(N, 0, -1):
        x = rk4_nl_adm(lambda x,y: L96_adm(x,y),lambda x: L96(x,F),zbase[:,i-1],x,dt)
    return x

sv_t  = 0.4  # 0.4 tu
sv_dt = 0.01 # 0.1 tu

sv = np.zeros((K,nmember,ncase))
for icase in range(ncase):
    print(f'case {icase+1} begin.')
    sv[:,:,icase] = singular_vectors(m=K, 
                      nsv=10, 
                      scale=3, 
                      tol=1e-6, 
                      P=P, 
                      r0=total_energy_norm(K),
                      rf=total_energy_norm(K),
                      TLM = lambda x: TLM(x,initial_state[:,0,icase],int(sv_t/sv_dt),sv_dt),
                      ADM = lambda x: ADM(x,initial_state[:,0,icase],int(sv_t/sv_dt),sv_dt),
                      nmember=50,
                      Da=Da[:,:,icase],
                      rescale=0.45)
## store
with open ('singular_vectors.pkl',"wb") as f:
    pickle.dump(sv,f)