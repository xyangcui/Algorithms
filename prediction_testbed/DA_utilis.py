import numpy as np
from scipy.linalg import inv

# predicted measurement
def h(x,m):
    '''
      Input:
        x: state vector.
        m: number of measurement.
      Output:
        Hx: predicted measurement.
    '''                    
    n = x.shape[0]           # dim of state vector
    H = np.zeros((m,n))
    d = int(n/m)
    for i in range(m):
        H[i,(i+1)*d-1] = 1

    return H @ x

# TLM operator of predicted measurement.
def Dh(x,m): 
    '''
      Input:
        x: state vector.
        m: number of measurement.
      Output:
        Hx: predicted measurement.
    '''                     
    n = x.shape[0]           
    H = np.zeros((m,n))
    d = int(n/m)
    for i in range(m):
        H[i,(i+1)*d-1] = 1

    return H

#### DA (Var / KF / nonlinear DA) ####
### with linear-gauss assumption ### 
## Variational method (iteration; not flow-dependent) ##
# 4DVar (constraint: whether model is perfect)
class FourDVar_practical:
    '''
      strongly constrained(SC)-4DVar
      Note
        1. It optimizes the solution at the beginning of an assimilation window.
        2. After that, one integrates the model solution to the end of the assimilation window with the deterministic nonlinear model,
          where the actual weather prediction starts.
        3. problem: almost all DA methods that optimize the start if an assimilation window:
              there is no guarantee that the solution will be the mode 
                after propagation to the end of the assimilation window where we initialize the actual forecasts.
      Procedures
        cost function: calculate cost.
        gradient func: calculate the gradient of cost function, currently use adjoint method. (may consider no adjoint method, like AutoDiff)
        optimizer: currently consider L-BFGS (the limited-memory version of Newton method).

      Definitions
        model: state evolution function.
        model_ADM: Adjoint model. 
        rk4: method, how to integrate.
        rk4_adm: method, how to integrate ADM.
      Reference:
        https://www.ecmwf.int/en/elibrary/79860-data-assimilation-concepts-and-methods
    '''

    def __init__(self,model_propagator,model_ADM_propagator):
        self.model_propagator = model_propagator
        self.model_ADM_propagator = model_ADM_propagator

    def _check_inputs(self, B, y, R, idx, N):
        idx = np.asarray(idx, dtype=int)
        R = np.asarray(R)
        if idx.ndim != 1:
            raise ValueError("idx must be a 1-D array of model-step observation indices")
        if len(idx) != y.shape[1]:
            raise ValueError(f"len(idx)={len(idx)} but y has {y.shape[1]} observation times")
        if len(idx) and (idx.min() < 0 or idx.max() > N):
            raise ValueError(f"observation indices {idx} must lie in [0, {N}]")
        if B.ndim == 2 and B.shape[0] == B.shape[1]:
            raise ValueError("B is a covariance matrix (square matrix); only vector-form B is accepted here")
        if R.ndim != 1:
            raise ValueError(f"R must be a 1-D vector, got shape {R.shape}")
        if R.shape[0] != y.shape[0]:
            raise ValueError(f"R length {R.shape[0]} != y length {y.shape[0]}")

    # cost function
    def cost_function(self,x,xb,B,R_inv,idx,y,h,N,lam,return_results=False):
        '''
          cost function of 4DVar
          type: J = 0.5*prior_error + 0.5*measure_error
          Input
             x: current state    (ndim)
            xb: prior state      (ndim)  
             B: background covariance vector (K,ndim)
             y: measurement / observation
             h: operational function h(x)
             N: time steps in an assimilation window.
          Output
             value of cost.
        '''
        from scipy.linalg import solve
        K = len(x)  # number of dims
        # step1: get values in an assimilation window
        x_traj = np.zeros((K,N+1)); x_traj[:,0] = x
        for i in range(N):
            x_traj[:,i+1] = self.model_propagator(x_traj[:,i])
        # step2: get value of prior error
        dim1, dim2 = B.shape
        dx = x - xb
        if dim1 == dim2:
           Jb = dx.T @ solve(B, dx)
        else:
            # use Woodbury (like the method used in ETKF)
            lam_1 = 1/lam
            term1 = lam_1*(dx.T@dx).item()
            term2 = lam*np.eye(dim1) + B@B.T
            By = B@dx
            term2By = solve(term2,By)
            term2_scalar = lam_1*((B.dot(dx)).T@term2By).item()
            Jb = term1 - term2_scalar 
        # step3: get value of measurement error
        Jo = 0.0
        for j, i in enumerate(idx):
            innovation = y[:, j] - h(x_traj[:, i])
            Jo += innovation.T @ R_inv @ innovation
        if return_results:
            return Jb, Jo
        else:
            return Jb + Jo

    # gradient for optimization.
    def gradient(self,x, xb, B, y, R_inv, idx, H, h, N, lam):
        '''
          adjoint-based method to calculate gradient.
          Input
            x: recent state.
            xb: background state.
            B:  background covariance matrix
            y: observation
            R_inv: observational covar
            H: observational operator
            h: observation function
            N: steps of assimilation window.
          Output
            gradient dimensions like x.
        '''
        from scipy.linalg import solve
        K = len(x)  #number of dim
        obs_map = {step: j for j, step in enumerate(idx)}
        # step1: get values in an assimilation window.
        x_traj = np.zeros((K,N+1)); x_traj[:,0] = x
        for i in range(N):
            x_traj[:,i+1] = self.model_propagator(x_traj[:,i])
        # step2: get gradient of observation.
        x_adj = np.zeros(K)
        for i in range(N, -1, -1):
            # step 1: calc forcing.
            if i in obs_map:
                j = obs_map[i]
                forcing = H(x_traj[:,i]).T @ R_inv @ (y[:,j] - h(x_traj[:,i]))
                x_adj = x_adj + forcing
            if i > 0:
                # step 2: integrate ADM. (practically, integrate TLM forward and ADM backward)
                x_adj = self.model_ADM_propagator(x_traj[:,i-1],x_adj)
        grad_Jo = -2 * x_adj
        #step3: get gradient of background. (B-1 dx)
        dim1, dim2 = B.shape
        dx = x-xb
        if dim1 == dim2:
            grad_Jb = 2 * solve(B, dx)
        else:
            # Woodbury
            lam_1 = 1/lam
            term1 = lam_1*dx
            term2 = lam*np.eye(dim1) + B@B.T
            By = B.dot(dx)
            term2By = solve(term2,By)
            term2_vector = lam_1*B.T@term2By
            grad_Jb = 2 * (term1-term2_vector)

        return grad_Jb + grad_Jo

    def four_dims_var_optimizer_scipy(
        self, xb, B, y, R, idx, n, H, h, lam=0.01, max_iter=300, tol=1e-3,
        verbose=True, return_history=False
    ):
        """Robust L-BFGS-B optimizer using the analytic adjoint gradient."""
        from scipy.optimize import minimize

        self._check_inputs(B, y, R, idx, n)
        R_inv = np.diag(1./R)
        # define a method to return cost. Input should one parameter.
        def fun(x):
            return self.cost_function(x, xb, B, R_inv, idx, y, h, n, lam)
        # define a method to return gradient. Input should one parameter.
        def jac(x):
            return self.gradient(x, xb, B, y, R_inv, idx, H, h, n, lam)
        
        result = minimize(
            fun, xb.copy(), jac=jac, method="L-BFGS-B",
            options={
                "maxiter": int(max_iter),
                "gtol": float(tol),
                "ftol": 1e-12,
                "maxls": 40,
                "maxcor": 10,
            },
        )

        if verbose:
            print(f"success: {result.success} ({result.message})")
            print(f"iterations: {result.nit}")
            print(f"gradient norm: {np.linalg.norm(result.jac):.6e}")
            print(f"initial cost: {fun(xb):.6e}")
            print(f"final cost: {fun(result.x):.6e}")

        if return_history:
            return result.x, result
        return result.x

class FourDVar_Incremental:
    '''
    Incremental 4DVar.
    It uses the thinking of Gaussian-Newton moethod.
    Firstly,  there is an outer loop: integrate a complex nonlinear model to get a trajectory. calculate distance d.
    Secondly, integrate a simple nonlinear model to get a trajectory for TLM and ADM.
    Thirdly, a inner loop to minimize cost function. The model is TLM.

    References

    '''
    # model info.
    def __init__(self,model_propagator,model_TLM_propagator,model_ADM_propagator,model_inner_propagator):
        self.model_propagator = model_propagator
        self.model_TLM_propagator = model_TLM_propagator
        self.model_ADM_propagator = model_ADM_propagator
        self.model_inner_propagator = model_inner_propagator

    def _check_inputs(self, B, y, R, idx, N):
        idx = np.asarray(idx, dtype=int)
        R = np.asarray(R)
        if idx.ndim != 1:
            raise ValueError("idx must be a 1-D array of model-step observation indices")
        if len(idx) != y.shape[1]:
            raise ValueError(f"len(idx)={len(idx)} but y has {y.shape[1]} observation times")
        if len(idx) and (idx.min() < 0 or idx.max() > N):
            raise ValueError(f"observation indices {idx} must lie in [0, {N}]")
        if B.ndim == 2 and B.shape[0] == B.shape[1]:
            raise ValueError("B is a covariance matrix (square matrix); only vector-form B is accepted here")
        if R.ndim != 1:
            raise ValueError(f"R must be a 1-D vector, got shape {R.shape}")
        if R.shape[0] != y.shape[0]:
            raise ValueError(f"R length {R.shape[0]} != y length {y.shape[0]}")
    # outer loop
    def outerloop(self,x_current,obs,idx,h):
        '''
            return nonliear state and y-h(x)
            Input
              x_current: current state at the start of assimilation window
              obs: observation [ndim,nobs].
              idx: which step has observation.
              h: nonlinear observation operator.
              N: length of DA window.
        '''
        # complex nonlinear model's trajectory
        x = self.model_propagator(x_current)
        # departure
        d = np.zeros_like(obs)
        for j, i in enumerate(idx):
            d[:,j] = obs[:,j] - h(x[:,i])
        return x, d
    ## inner loop
    # cost function
    def cost_function(self,x,xl_traj,xh_traj,d,B,R_inv,idx,H,lam,return_results=False):
        '''
          cost function of 4DVar
          type: J = 0.5*prior_error + 0.5*measure_error
          Input
             x: current state    (ndim)
            xl_traj: simple nonlinear model's trajectory (ndim,N)
            xh_traj: complex nonlinear model's trajectory
             d: distance. (ndim, nobs)  
             B: background covariance vector (K,ndim)
             y: measurement / observation
             h: operational function h(x)
             N: time steps in an assimilation window.
          Output
             value of cost.
        '''
        from scipy.linalg import solve
        K = len(x)  # number of dims
        # step1: get values in an assimilation window
        x_traj = self.model_TLM_propagator(x,xl_traj)
        # step2: get value of prior error
        dim1, dim2 = B.shape
        dx = x
        if dim1 == dim2:
           Jb = dx.T @ solve(B, dx)
        else:
            # use Woodbury (like the method used in ETKF)
            lam_1 = 1/lam
            term1 = lam_1*(dx.T@dx).item()
            term2 = lam*np.eye(dim1) + B@B.T
            By = B@dx
            term2By = solve(term2,By)
            term2_scalar = lam_1*((B.dot(dx)).T@term2By).item()
            Jb = term1 - term2_scalar 
        # step3: get value of measurement error
        Jo = 0.0
        for j, i in enumerate(idx):
            innovation = d[:,j] - H(xh_traj[:,i])@x_traj[:,i]
            Jo += innovation.T @ R_inv @ innovation
        if return_results:
            return Jb, Jo
        else:
            return Jb + Jo
    # gradient of cost function
    def gradient(self,x,xl_traj, d, B, R_inv, idx, H, N, lam):
        '''
          adjoint-based method to calculate gradient.
          Input
            x: recent state.
            xb: background state.
            B:  many realizations that stores info of background matrix.
            y: observation
            R_inv: observational covar
            H: observational operator
            h: observation function
            N: steps of assimilation window.
          Output
            gradient dimensions like x.
        '''
        from scipy.linalg import solve
        K = len(x)  #number of dim
        obs_map = {step: j for j, step in enumerate(idx)}
        # step1: get values in an assimilation window.
        x_traj = self.model_TLM_propagator(x,xl_traj)
        # step2: get gradient of observation.
        x_adj = np.zeros(K)
        for i in range(N, -1, -1):
            # step 1: calc forcing.
            if i in obs_map:
                H_info = H(xl_traj[:,i])
                j = obs_map[i]
                forcing = H_info.T @ R_inv @ (d[:,j] - H_info@x_traj[:,i])
                x_adj = x_adj + forcing
            if i > 0:
                # step 2: integrate ADM. (practically, integrate TLM forward and ADM backward)
                x_adj = self.model_ADM_propagator(xl_traj[:,i-1],x_adj)
        grad_Jo = -2 * x_adj
        #step3: get gradient of background. (B-1 dx)
        dim1, dim2 = B.shape
        dx = x
        if dim1 == dim2:
            grad_Jb = 2 * solve(B, dx)
        else:
            # Woodbury
            lam_1 = 1/lam
            term1 = lam_1*dx
            term2 = lam*np.eye(dim1) + B@B.T
            By = B.dot(dx)
            term2By = solve(term2,By)
            term2_vector = lam_1*B.T@term2By
            grad_Jb = 2 * (term1-term2_vector)

        return grad_Jb + grad_Jo    
    # optimizer
    def four_dims_var_optimizer_scipy(
        self, x_start, xl_traj, xh_traj, d, B, R, idx, N, H, lam, max_iter, tol,
        verbose, return_history
    ):
        """Robust L-BFGS-B optimizer using the analytic adjoint gradient."""
        from scipy.optimize import minimize
        R_inv = np.diag(1./R)
        # define a method to return cost. Input should one parameter.
        def fun(x):
            return self.cost_function(x,xl_traj,xh_traj,d,B,R_inv,idx,H,lam,return_results=False)
        # define a method to return gradient. Input should one parameter.
        def jac(x):
            return self.gradient(x,xl_traj, d, B, R_inv, idx, H, N, lam)

        result = minimize(
            fun, x_start.copy(), jac=jac, method="L-BFGS-B",
            options={
                "maxiter": int(max_iter),
                "gtol": float(tol),
                "ftol": 1e-12,
                "maxls":  40,
                "maxcor": 10,
            },
        )

        if verbose:
            print(f"success: {result.success} ({result.message})")
            print(f"iterations: {result.nit}")
            print(f"gradient norm: {np.linalg.norm(result.jac):.6e}")
            print(f"initial cost: {fun(x_start):.6e}")
            print(f"final cost: {fun(result.x):.6e}")

        if return_history:
            return result.x, result
        return result.x
    
    # core function
    def core_procedure(
        self, xb, B, y, R, idx, N, h, H, S, S_inv, fcst_step, n_outer=4, lam=0.01, max_iter=300, tol=1e-3,
        verbose=False, return_history=False
    ):
        '''
        Input
            xb: complex model background state.
             B: many realizations that stores info of background matrix.
             y: measurement/observation.
             idx: step that has observation.
             S: map from complex to simple model
             S_inv: map from simple to complex model.
             fcst_step: distance from start of DA window to do forecast.
             n_outer: integrating loop of outer step.
             lam: for regularization.
             max_iter: maximum iteration
        Outout
            xa: analysis state.
        '''
        self._check_inputs(B, y, R, idx, N)
        x_current = xb.copy()
        for i in range(n_outer):
            # integrate outer loop.
            x_outer, d_outer = self.outerloop(x_current,y,idx,h)
            # integrate simple nonlienar model.
            x_inner0 = S(x_current)
            x_inner = self.model_inner_propagator(x_inner0)
            K = x_inner0.size       
            # initial increment
            if i == 0:
                x_start = np.zeros(K)
            else:
                x_start = xb - x_current
            # inner loop
            dx = self.four_dims_var_optimizer_scipy(
                                                    x_start,
                                                    x_inner,
                                                    x_outer,
                                                    d_outer,
                                                    B,
                                                    R,
                                                    idx,
                                                    N,
                                                    H,
                                                    lam=lam,
                                                    max_iter=max_iter,
                                                    tol=tol,
                                                    verbose=verbose,
                                                    return_history=return_history)
            # add to the final
            if i == n_outer-1:
                '''Late 4D-start to generate an-type fields'''
                x_inc = self.model_TLM_propagator(dx,x_inner)
                x_an  = x_outer[:,fcst_step] + S_inv(x_inc[:,fcst_step])
            else:
                x_current = x_current + S_inv(dx)

        return x_an

## Ensemble Kalman Filter  (assumption: model is equivalent to linear model; flow-dependent) ##
# localization
#G-C function
def comp_cov_factor(z_in,c):
    '''
    Input
      z_in: distance between two points.
      c: cutoff radius.
    Output
      cov_factor: a localization coefficient 
    '''
    z = abs(z_in)
    if z <= c:
        r = z/c
        cov_factor = -0.25*r**5+0.5*r**4+0.625*r**3-5.0/3.0*r**2 + 1
    elif z <= 2*c:
        r = z/c
        cov_factor = 1.0/12.0*r**5-0.5*r**4+0.625*r**3-5.0/3.0*r**2\
            -5.0*r+4-2.0/(3.0*r)
    else:
        cov_factor = 0

    return cov_factor

def Rho_theretical(localP, size):
    '''
    An easy version to generate a localized matrix.
    It may include real distance in the future.
    Input
      localP: cutoff radius
      size: length of a vector
    Output
      a matrix.
    '''
    from scipy.linalg import toeplitz
    rho0 = np.zeros(size)
    for i in range(size):
        rho0[i] = comp_cov_factor(i,localP)

    return toeplitz(rho0,rho0)
def Rho(localP, dist):
    '''
    An easy version to generate a localized matrix.
    It may include real distance in the future.
    Input
      localP: cutoff radius
      dist[ndim,nobs]: distance between point and obs.
    Output
      a matrix.
    '''
    rho = np.zeros_like(dist)
    ndim, nobs = dist.shape
    for i in range(ndim):
        for j in range(nobs):
            rho[i,j] = comp_cov_factor(dist[i,j],localP)

    return rho
# EnKF
def enkf_update_array(xb,y,ObsOp,R,gamma=1.,loc=None):
    '''
    Ensemble Kalman Filter
    Input 
      xb: prior estimate (ndim,nens)
       y: measurement (nobs)
      ObsOp: projection model.
       R: covariance of measurement (nobs,nobs)
    gamma: parameter of inflation. default is 1, no inflation.
     loc: localization matrix. Default is no localization. (ndim, nobs)
    Output
      xa: posterior estimate (ndim,nens)

    Reference
      Evensen, G., F. C. Vossepoel, and P. J. Van Leeuwen, 2022: 
        Data Assimilation Fundamentals: A Unified Formulation of the State and Parameter Estimation Problem. 
        Springer International Publishing, https://doi.org/10.1007/978-3-030-96709-3.
    '''
    from math import sqrt
    # step1: dim information
    _, nens = xb.shape; nobs = len(y)
    # step2: get centralized matrix
    IN = np.eye(nens); I = np.ones(nens)
    PI = (IN - np.outer(I,I)/nens)/np.sqrt(nens-1)
    # step3: disturb measurement
    E  = np.random.multivariate_normal(np.zeros(nobs), R, size=nens).T #[nobs, nens]
    D  = y[:,None]@I[None,:] + np.sqrt(nens-1)*E
    # step4: project prior to measurement
    y_model  = ObsOp(xb); Y = y_model@PI; X = xb@PI
    # step5 : update
    if loc is None:
        W  = Y.T@inv(Y@Y.T+E@E.T)@(D - y_model)
    else:
        W  = loc*Y.T@inv(Y@Y.T+E@E.T)@(D - y_model)
    ## inflation
    xm = xb.mean(axis=1)
    xbp = xb - xm; xbp *= sqrt(gamma)
    xb = (xm[:,None]+xbp) @ (I + W/np.sqrt(nens-1))  

    return xb

# square-root Filters
# LETKF
def letkf_update_array(E,R,y,H,loc,gamma=1.0):
    '''
    Local Ensemble Transform Kalman Filter
    Input
      E: prior estimation (ndims, nens)
      R: covariance matrix of measurements (dim_measure,dim_measure) independent
      y: measurement (dim_measurement)
      H: corelation between model and measurement, linear case (dim_measure,ndims)
      gamma: parameter for inflation. default is 1.0, no inflation
     loc: localization matrix. Default is no localization. (ndim, nobs)
    Output
      E: posterior estimation (ndims, nens)

    Referfence: Hunt, B. R., E. J. Kostelich, and I. Szunyogh, 2007: 
      Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. 
      Physica D: Nonlinear Phenomena, 230, 112–126, https://doi.org/10.1016/j.physd.2006.11.008.
    '''
    from scipy.linalg import solve
    D, nens = E.shape[0], E.shape[1]
    # seperate prior into mean and anomaly
    xbb = np.nanmean(E,1).reshape(D,1)
    xbp = E - xbb #/ np.sqrt(nens - 1)
    # model to measurement
    Y  = H(E)
    ym = np.nanmean(Y,1); yp = Y - ym[:, None]
    innovation = y - ym
    # update
    for i in range(D):
        #solve RC = yb get R-1yp transpose: yp.T@R-1
        rho = loc[i, :]
        C = solve(R, yp, assume_a='pos'); CT = C.T
        CT_loc = CT * rho[None, :]
        A_mat = (nens-1)/gamma* np.eye(nens) + CT_loc @ yp
        # use PCA get Pa.
        eigvals, eigvecs = np.linalg.eigh(A_mat)
        tol = 1e-8 * np.max(eigvals)
        eigvals = np.where(eigvals < tol, tol, eigvals)
        Pa = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        if not np.allclose(Pa, Pa.T):
            Pa = (Pa + Pa.T) / 2
            eigvals_p, eigvecs_p = np.linalg.eigh((nens-1)*Pa)
        else:
            eigvals_p = (nens-1) * 1.0 / eigvals
            eigvecs_p = eigvecs

        eigvals_p = np.maximum(eigvals_p, 0.0) 
        # calculate W and w.
        W = eigvecs_p @ np.diag(np.sqrt(eigvals_p)) @ eigvecs_p.T
        w = Pa @ CT_loc @ innovation[:, None]
        if np.iscomplexobj(W):
            W = np.real(W)
        # update locally
        E[i, :] = xbb[i, 0] + xbp[i, :] @ (w + W)

    return E

# sequential EnKS based on LETKF
def enks_letkf_update_array(E,E0,R,y,H,loc,gamma=1.0):
    '''
    Kalman smoother based LETKF.
    Input
      E: prior estimation (ndims, nens)
      R: covariance matrix of measurements (dim_measure,dim_measure) independent
      y: measurement (dim_measurement)
      H: corelation between model and measurement, linear case (dim_measure,ndims)
      gamma: parameter for inflation. default is 1.0, no inflation
     loc: localization matrix. Default is no localization. (ndim, nobs)
    Output
      E: posterior estimation (ndims, nens)
      E0update: 
    '''
    from scipy.linalg import solve
    D, nens = E.shape[0], E.shape[1]
    # seperate prior into mean and anomaly
    xbb = np.nanmean(E,1).reshape(D,1)
    xbp = E - xbb #/ np.sqrt(nens - 1)
    # smoother mean.
    E0mean = np.nanmean(E0, axis=1, keepdims=True)
    E0anom = E0 - E0mean
    # model to measurement
    Y  = H(E)
    ym = np.nanmean(Y,1); yp = Y - ym[:, None]

    C = solve(R, yp, assume_a='pos'); CT = C.T
    innovation = y - ym

    # Inner function: local LETKF transform
    def get_local_transform(i):

        rho = loc[i, :]
        CT_loc = CT * rho[None, :]
        A_mat = (
            (nens - 1) / gamma * np.eye(nens)
            + CT_loc @ yp
        )

        eigvals, eigvecs = np.linalg.eigh(A_mat)

        tol = 1e-8 * np.max(eigvals)
        eigvals = np.maximum(eigvals, tol)

        # Pa in ensemble space
        Pa = (
            eigvecs
            @ np.diag(1.0 / eigvals)
            @ eigvecs.T
        )

        # perturbation transform
        W = (
            eigvecs
            @ np.diag(np.sqrt((nens - 1) / eigvals))
            @ eigvecs.T
        )

        # mean weights
        w = Pa @ CT_loc @ innovation[:,None]

        return w + W
    
    # update
    if E0.ndim == 2:
        for i in range(D):
            trans_matrix = get_local_transform(i)
            E[i, :] = xbb[i, 0] + xbp[i, :] @ trans_matrix
            # smoother
            E0[i,:] = E0mean[i, 0] + E0anom[i, :] @ trans_matrix
    else:
        for i in range(D):
            trans_matrix = get_local_transform(i)
            E[i, :] = xbb[i, 0] + xbp[i, :] @ trans_matrix            
            E0[i,:,:] = E0mean[i, 0, :][None, :] + np.einsum('dec,ef->dfc',E0anom[i, :, :],trans_matrix)

    return E

# ETKF
def etkf_update_array_theory(E,R,R_inv,y,H,gamma=1.0):
    '''
      Ensemble Transform Kalman Filter
      Input
        E: prior estimation (ndims, nens)
        R: covariance matrix of measurements (dim_measure,dim_measure)
        R_inv: inversed covariance matrix of measurements (dim_measure,dim_measure)
        y: measurement (dim_measurement)
        H: corelation between model and measurement, linear case (dim_measure,ndims)
        gamma: parameter for inflation. default is 1.0, no inflation

      Output
        posterior estimation (ndims, nens)

      Reference: Bishop et al. 2001 MWR
        Bishop, C. H., B. Etherton, and S. J. Majumdar, 2001: Adaptive sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects. 
        Mon. Wea. Rev., 129, 420–436. https://doi.org/10.1175/1520-0493(2001)129<0420:ASWTET>2.0.CO;2
    '''
    from math import sqrt
    D, nens = E.shape
    # step1: get mean and anomaly.
    x_mean = np.mean(E,axis=1)
    x_anom = np.subtract(E,x_mean[:,np.newaxis])
    # get mean analysis state vector
    Pf  = np.dot(x_anom,x_anom.T)/(nens-1)
    tmp = H @ Pf @ H.T + R
    # step2: innovate mean state at first.
    if np.linalg.matrix_rank(tmp) == tmp.shape[0] :
        K1 = np.linalg.inv(tmp)
    else:
        print("The martrix is not fully ranked, use pseudo inv.")
        K1 = np.linalg.pinv(tmp)  #pseudo inv
    K = Pf @ H.T @ K1
    innov = y - H @ x_mean
    xam = x_mean + np.dot(K, innov)
    #step3: innovate anomalous state.
    # get square root.
    zf = x_anom/np.sqrt(nens-1)
    tmp = zf.T @ H.T @ R_inv @ H @ zf
    # ensure C is symmetric.
    C = (tmp + tmp.T)/2
    # EVD for C.
    eigvalues, eigvectors = np.linalg.eigh(C)
    # get analysis square root
    T = eigvectors @ np.diag(1.0 / np.sqrt(1. + eigvalues))
    # innovate perturbation.
    E = xam[:,np.newaxis] + sqrt(gamma) * (x_anom @ T)

    return E

#ETKF
def etkf_update_array(E,R,y,H,gamma=1.0):
    '''
    Ensemble Transform Kalman Filter
    Input
      E: prior estimation (ndims, nens)
      R: covariance matrix of measurements (dim_measure,dim_measure)
      y: measurement (dim_measurement)
      H: corelation between model and measurement, linear case (dim_measure,ndims)
      gamma: parameter for inflation. default is 1.0, no inflation
    Output
      E: posterior estimation (ndims, nens)

    Referfence: Hunt, B. R., E. J. Kostelich, and I. Szunyogh, 2007: 
      Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. 
      Physica D: Nonlinear Phenomena, 230, 112–126, https://doi.org/10.1016/j.physd.2006.11.008.
    '''
    from scipy.linalg import solve
    D, nens = E.shape[0], E.shape[1]
    # seperate prior into mean and anomaly
    xbb = np.nanmean(E,1).reshape(D,1)
    xbp = E - xbb #/ np.sqrt(nens - 1)
    # model to measurement
    Y  = H(E)
    ym = np.nanmean(Y,1); yp = Y - ym[:, None]
    innovation = y - ym
    #solve RC = yb get R-1yp transpose: yp.T@R-1
    C = solve(R, yp, assume_a='pos'); CT = C.T
    A_mat = (nens-1)/gamma* np.eye(nens) + CT @ yp
    # use PCA get Pa.
    eigvals, eigvecs = np.linalg.eigh(A_mat)
    tol = 1e-8 * np.max(eigvals)
    eigvals = np.where(eigvals < tol, tol, eigvals)
    Pa = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    if not np.allclose(Pa, Pa.T):
        Pa = (Pa + Pa.T) / 2
        eigvals_p, eigvecs_p = np.linalg.eigh((nens-1)*Pa)
    else:
        eigvals_p = (nens-1) * 1.0 / eigvals
        eigvecs_p = eigvecs

    eigvals_p = np.maximum(eigvals_p, 0.0) 
    # calculate W and w.
    W = eigvecs_p @ np.diag(np.sqrt(eigvals_p)) @ eigvecs_p.T
    w = Pa @ CT @ innovation[:,None]
    if np.iscomplexobj(W):
        W = np.real(W)
    # udate
    E = xbb+xbp@(w+W)
    return E

# sequential EnKS (based on etkf).
def enks_etkf_update_array(E,E0,R,y,H,gamma=1.0):
    '''
    Kalman smoother based on ETKF.
    Input
      E: prior estimation (ndims, nens)
      E0: state wait for update. (ndims, nens, ncase)
      R: covariance matrix of measurements (dim_measure,dim_measure)
      y: measurement (dim_measurement)
      H: corelation between model and measurement, linear case (dim_measure,ndims)
      gamma: parameter for inflation. default is 1.0, no inflation
    Output
      E: posterior estimation (ndims, nens)
      E0_update: 
    '''
    from scipy.linalg import solve
    D, nens = E.shape[0], E.shape[1]
    # seperate prior into mean and anomaly
    xbb = np.nanmean(E,1).reshape(D,1)
    xbp = E - xbb #/ np.sqrt(nens - 1)
    # model to measurement
    Y  = H(E)
    ym = np.nanmean(Y,1); yp = Y - ym[:, None]
    innovation = y - ym
    #solve RC = yb get R-1yp transpose: yp.T@R-1
    C = solve(R, yp, assume_a='pos'); CT = C.T
    A_mat = (nens-1)/gamma* np.eye(nens) + CT @ yp
    # use PCA get Pa.
    eigvals, eigvecs = np.linalg.eigh(A_mat)
    tol = 1e-8 * np.max(eigvals)
    eigvals = np.where(eigvals < tol, tol, eigvals)
    Pa = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    if not np.allclose(Pa, Pa.T):
        Pa = (Pa + Pa.T) / 2
        eigvals_p, eigvecs_p = np.linalg.eigh((nens-1)*Pa)
    else:
        eigvals_p = (nens-1) * 1.0 / eigvals
        eigvecs_p = eigvecs

    eigvals_p = np.maximum(eigvals_p, 0.0) 
    # calculate W and w.
    W = eigvecs_p @ np.diag(np.sqrt(eigvals_p)) @ eigvecs_p.T
    w = Pa @ CT @ innovation[:,None]
    if np.iscomplexobj(W):
        W = np.real(W)
    # udate
    trans_matrix = w + W
    E = xbb+xbp@trans_matrix
    # smoother.
    E0mean = np.nanmean(E0, axis=1, keepdims=True)
    E0anom = E0 - E0mean

    if E0.ndim == 2:
        E0update = E0mean + E0anom @ trans_matrix
    else:
        E0update = E0mean + np.einsum('dec,ef->dfc',E0anom,trans_matrix)

    return E, E0update
# EAKF

# serial SRF
def serial_update_array(E,R,obs,h,gamma=1.0,loc=None):
    '''
      Ensemble serial root-squre filter.

      Input
        E: prior estimation (ndims, nens)
        R: covariance matrix of measurements (dim_measure,dim_measure)
        obs: measurement (dim_measurement)
        h: projection operator (dim_measure,nens)
        gamma: parameter for inflation. default is 1.0, no inflation
        loc: localization
      Output
        E; posterior estimation (ndims, nens)
      
      Reference: Whitaker and Hamill 2002 MWR
        Whitaker, J., and T. M. Hamill, 2002: Ensemble data assimilation without perturbed observations. 
        Mon. Wea. Rev., 130, 19131924. https://doi.org/10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2
    '''
    from math import sqrt
    # dim numbers and projection
    nens = E.shape[0]; nmea = len(obs); Y = h(E)
    for i in range(nmea):
        y = Y[i,:]; r = R[i,i]; y_obs = obs[i]
        # step1: seperate ensemble mean and anomaly.
        xm = np.mean(E,axis=1)
        xf = np.subtract(E,xm[:,np.newaxis])
        # seperate measurement mean and anomaly.
        ym = np.mean(y,axis=0)
        yf = np.subtract(y,ym)
        # step2: variance of measurement.
        varye = np.var(yf,ddof=1)
        # covariance between state and measurement.
        covxy = np.dot(xf,yf)/(nens-1)
        # step3: kalman gain.
        K = np.divide(covxy,r+varye)
        # beta coefficient
        beta = 1./(1.+np.sqrt(r/(r+varye))) 
        # innovation
        xam = xm + np.multiply(K,y_obs-ym)
        # convert to martrix
        if loc == None:
            Kmat = np.multiply(beta,K)
        else:
            Kmat = loc[:,i]@np.multiply(beta,K)
        xa = xf - np.dot(Kmat[:,np.newaxis],yf[np.newaxis,:])
        E = xam[:,np.newaxis] + sqrt(gamma)*xa

    return E

### without linear-gauss assumption (fully nonlinear filter) ###

# particle filter

# particle-flow filter

### coupled data assimilation ###
## strong CDA
## weak CDA