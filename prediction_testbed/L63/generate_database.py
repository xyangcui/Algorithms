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
# create psedo observation.
obs_std = 1.5  # observation-error standard deviation
np.random.seed(42)   # for reproducibility
obs_noise = np.random.normal(loc=0.0,scale=obs_std,size=X.shape)
Y = X + obs_noise

# observation matrix
R = np.eye(K) * obs_std*obs_std
# background matrix
B = np.cov(X)

data = {'real': X, 'B': B, 'obs_std': obs_std, 'obs': Y, 'R': R}

# store value
with open('database.pkl', 'wb') as f:
    pickle.dump(data, f)