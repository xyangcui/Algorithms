import numpy as np
import matplotlib.pyplot as plt
from fun_dynsys_univariate_analysis import fun_dynsys_univariate_analysis
"""
%INFO%
%Example of the computation of local dimensions and persistence for the
%Lorenz 1963 equations


%REFERENCES%
%Please cite:

%Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
%drivers of weather extremes: application to hot and cold days in North 
%America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

%Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
%of North Atlantic predictability and extremes. Scientific Reports, 7, 
%41278, doi: 10.1038/srep41278 


%INPUT%
%quanti: Percentile determining the radius of the sphere in the space of
%distances.


%OUTPUTS%
%From this code we extract the following time series:
%Inverse persistence theta
%Local dimensions D1
%They both have dimension [time]
"""
## INPUTS
quanti = 0.98;

## LORENZ ATTRACTOR SIMULATION
# Define the integration total time T and the time-step dt;
T    = 100000
dt   = 0.01
t    = np.arange(dt, T)

# Lorenz attractor parameter
beta  = 8/3 
sigma = 10 
rho   = 28
# Lorenz Initial Conditions

# INITIALIZATION AND INTEGRATION
x = np.zeros(T)
y = np.zeros(T)
z = np.zeros(T)

x[0] = 1.
y[0] = 1.
z[0] = 1.

# Iterate Lorenz attractor with an Euler Scheme
for i in range(T-2):
    x[i+1] = x[i] + dt*(sigma*(y[i]-x[i]) )
    y[i+1] = y[i] + dt*(x[i]*(rho - z[i])- y[i]) 
    z[i+1] = z[i] + dt*(x[i]*y[i] - beta*z[i]) 


## DIMENSION AND PERSISTENCE COMPUTATION
# rearrange the trajectory in a single matrix [TIMExSPACE]
data = np.column_stack((x, y, z))

D1, theta = fun_dynsys_univariate_analysis(data,quanti)

## EXAMPLE PLOTS
# Plot a section of the Lorenz attractor with D1 and theta in colors

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
fig.suptitle('Lorenz Attractor and its Dimensions')
axs[0].scatter(x, z, c=D1, cmap='rainbow', alpha=0.6)
axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[0].set_title('Local Dimensions')
axs[1].scatter(x, z, c=theta, cmap='viridis', alpha=0.6)
axs[1].set_xlabel('x')
axs[1].set_ylabel('z')
axs[1].set_title('Local Inverse Persistence')
plt.show()