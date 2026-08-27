import numpy as np
from math import sin

# Runge-Kuta integration
def RK4(rhs,state,dt,*args):
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2.,*args)
    k3 = rhs(state+k2*dt/2.,*args)
    k4 = rhs(state+k3*dt,*args)

    return state+(k1+2*k2+2*k3+k4)*dt/6.

# Lorenz 63 model.
def L63(state,*args):
    sigma = args[0]  #params
    rho   = args[1]
    beta  = args[2]

    x,y,z = state

    f = np.zeros(3)
    f[0] = sigma*(y-x)
    f[1] = rho*x-y-x*z  #rho heating parameters
    f[2] = x*y - beta*z

    return f

# L63 TLM operator
def L63_TLM_operator(state,dt,*args):
    #def L.
    sigma = args[0]  #params
    rho   = args[1]
    beta  = args[2]

    x,y,z = state

    J = np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]])

    L = np.eye(len(state))

    def operator(M):
        return J@M

    return RK4(operator,L,dt)

# modified Lorenz 63 model. add periodicity. 
def L63_modified(state,sigma,r0,r1,beta,T,I):

    x,y,z,p = state

    f = np.zeros(4)
    f[0] = sigma*(y-x)
    f[1] = (r0+r1*sin(p))*x-y-x*z  
    f[2] = x*y - beta*z
    f[3] = 1/T

    return f

# Lorenz 96 model.
def L96(state,*args):

    x = state
    F = args[0]         #Forcing
    n = len(state)      #dims
    f = np.zeros(n,dtype=np.float64)  

    if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e10):
        return np.zeros(n)
    
    #boundary
    f[0] = (x[1] - x[n-2])* x[n-1] - x[0]
    f[1] = (x[2]- x[n-1])* x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]

    #inner
    for i in range(2,n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]

    return f+F

# L96 TLM propagator
def L96_TLM_operator(x,dt):
    #def L.
    K = x.shape[0]
    J = np.zeros((K,K))

    for k in range(K):
        J[k, (k-2) % K] = -x[(k-1) % K]  # -X_{k-1}
        J[k, (k-1) % K] = -x[(k-2) % K] + x[(k+1) % K]  # -X_{k-2} + X_{k+1}
        J[k, k] = -1.0  # damping
        J[k, (k+1) % K] = x[(k-1) % K]  # X_{k-1}

    L = np.eye(len(x))

    def operator(M):
        return J@M

    return RK4(operator,L,dt)

# Lorenz 96 model. version: with small-scale dynamics.
def L96_couple(state, n, F, c, b, h, z_ref):
 
    J, n_z = z_ref.shape  
    
    x = state[:n]
    z_flat = state[n:]
    z = z_flat.reshape(J, n_z)

    if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e10):
        return np.zeros(n)

    #large scale.
    dxdt = np.zeros(n)
    dxdt[0] = (x[1] - x[n-2]) * x[n-1] - x[0]
    dxdt[1] = (x[2] - x[n-1]) * x[0] - x[1]
    dxdt[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    for i in range(2, n-1):
        dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    
    dxdt += F  
    
    coupling = h * c / b * np.mean(z, axis=0)
    dxdt -= coupling
    
    # small scale.
    dzdt = np.zeros_like(z)
    dzdt[0,:] = c*b*z[1,:]*(z[J-1,:]-z[2])*(-1) - c*z[0,:] 
    dzdt[J-2,:] = c*b*z[J-1,:]*(z[J-3,:]-z[0,:])*(-1) - c*z[J-2,:] 
    dzdt[J-1,:] = c*b*z[0,:]*(z[J-2,:]-z[1,:])*(-1) - c*z[J-1,:] 

    for j in range(1,J-2):
        dzdt[j,:] = c*b*z[j+1,:]*(z[j-1,:]-z[j+2,:])*(-1) - c*z[j,:] 

    dzdt += h * c / b * x
    
    return np.concatenate([dxdt, dzdt.flatten()])

# parameterized version of L2S model.
def L96_para(state,*args):

    x = state
    F = args[0]         #Forcing
    n = len(state)      #dims
    f = np.zeros(n,dtype=np.float64)  

    if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e10):
        return np.zeros(n)
    
    #boundary
    f[0] = (x[1] - x[n-2])* x[n-1] - x[0]
    f[1] = (x[2]- x[n-1])* x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]

    #inner
    for i in range(2,n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]

    gu = -1.31*x-0.27
    return f+F-gu