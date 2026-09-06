import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96
from solve_ode import runge_kuta4
import numpy as np
import pickle

K  = 40                  # scale
F  = 8.                  # Forcing

dt = 0.05
tu = 1000
nt = int(tu/dt)

# spin-up
x0 = np.repeat(F,K)
x0[int(K/2)] += 0.01
for k in range(int(100/dt)):
    x0 = runge_kuta4(lambda x: L96(x,F),x0,dt)

# formal integration
X = np.zeros((K,nt+1),dtype=np.float64)
X[:,0] = x0
for j in range(nt):
    X[:,j+1] = runge_kuta4(lambda x: L96(x,F),X[:,j],dt)
# background matrix
B = np.cov(X)
reg_param = 1e-6 * np.trace(B) / K
B += reg_param * np.eye(K)
# create psedo observation.
# observation matrix
R = np.diag(np.diag(B))
np.random.seed(42)   # for reproducibility
obs_noise = np.random.multivariate_normal(mean=np.zeros(K),cov=R,size=X.shape[1])
Y = X + obs_noise.T

data = {'real': X, 'B': B, 'obs': Y, 'R': R}

# store value
with open('database.pkl', 'wb') as f:
    pickle.dump(data, f)