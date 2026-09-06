import numpy as np
from scipy.linalg import sqrtm, inv

def h(x, obs_ratio=1.):
    K = x.shape[0]

    step = max(1, int(round(1.0 / obs_ratio)))
    obs_idx = np.arange(0, K, step)

    Hx = np.zeros(K)
    Hx[obs_idx] = x[obs_idx]

    return Hx


def Dh(x, obs_ratio=1.):
    K = x.shape[0]

    step = max(1, int(round(1.0 / obs_ratio)))
    obs_idx = np.arange(0, K, step)

    H = np.zeros((K, K))
    H[obs_idx, obs_idx] = 1.0

    return H

#### DA (Var / KF / nonlinear DA) ####
### with linear-gauss assumption ### 
## Variational method (iteration; not flow-dependent) ##
# 4DVar (constraint: whether model is perfect)
class FourDVar:
    '''
      strongly constrained(SC)-4DVar
      an easy algorithm because prediction step must equal to observation's.

      Definitions
        model: state evolution function.
        model_TLM: tangent linear model. 
        rk4: method, how to integrate.

      Reference:
        https://www.ecmwf.int/en/elibrary/79860-data-assimilation-concepts-and-methods
    '''

    def __init__(self,model,model_TLM,rk4):
        self.model = model
        self.model_TLM = model_TLM
        self.RK4 = rk4

    # cost function
    def cost_function(self,x,xb,B_inv,R_inv,y,h,N):
        '''
          cost function of 4DVar
          type: J = 0.5*prior_error + 0.5*measure_error
          Input
             x: current state    (ndim)
            xb: prior state      (ndim)  
            B_inv: inversed background covariance matrix (ndim,ndim)
             y: measurement / observation
             h: operational function h(x)
             N: time steps in an assimilation window
          Output
             value of cost.
        '''
        K = len(x)  # number of dims
        # step1: get values in an assimilation window
        x_traj = np.zeros((K,N)); x_traj[:,0] = x
        for i in range(N-1):
            x_traj[:,i+1] = self.RK4(self.model,x_traj[:,i])
        # step2: get value of prior error
        dx = x - xb
        Jb = dx.T @ B_inv @ dx
        # step3: get value of measurement error
        Jo = 0.0
        for i in range(N):
            innovation = y[:, i] - h(x_traj[:, i],K)
            Jo += innovation.T @ R_inv @ innovation
            
        return Jb + Jo

    # gradient for optimization.
    def gradient(self,x, xb, B_inv, y, R_inv, H, h, N):
        '''
          adjoint-based method to calculate gradient.
          Input
            x: recent state.
            xb: background state.
            B_inv: inverted background covariance matrix
            y: observation
            R_inv: observational covar
            H: observational operator
            h: observation function
            N: times of observation.
          Output
            gradient dimensions like x.
        '''
        K = len(x)  #number of dim
        # step1: get values in an assimilation window.
        x_traj = np.zeros((K,N)); x_traj[:,0] = x
        for i in range(N-1):
            x_traj[:,i+1] = self.RK4(self.model,x_traj[:,i])
        # step2: get gradient of observation.
        x_adj = np.zeros(K)
        for i in range(N-1, -1, -1):
            # step 1: calc forcing.
            forcing = H(x_traj[:,i],K).T @ R_inv @ (y[:,i] - h(x_traj[:,i],K))
            x_adj = x_adj + forcing
            if i>0:
                # step 2: integrate ADM. (practically, integrate TLM forward and ADM backward)
                M = self.model_TLM(x_traj[:,i-1])
                x_adj = M.T @ x_adj
        # observation contribution at t_0
        #forcing = H(x_traj[:, 0], K).T@ R_inv@ (y[:, 0] - h(x_traj[:, 0], K))   
        #x_adj += forcing
        grad_Jo = -2 * x_adj
        #step3: get gradient of background.
        grad_Jb = 2* B_inv @ (x-xb)

        return grad_Jb + grad_Jo

    # optimizer quais Newton BFGS.
    def LBFGS_update(self,grad,s_list,y_list):
        '''
        unconstrained optimization algorithm.
        quasi Newton method. limited-BFGS. for optimal minimum problem.
        if q = -grad, minimum; else q = grad, maximum
        reference: https://zhuanlan.zhihu.com/p/514576143
        Input
          grad: gradient f(x_k)
          s_list: list retained vector s. sequence: early to late
          y_list: list retained vector y.
        Output
          r: increment. a vector
      '''
        # loop-1: reverse traversal.
        q = -grad; alpha_list = []
        for i in reversed(range(len(s_list))):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / np.dot(y.T, s)
            alpha = rho*np.dot(s.T,q)
            q -= alpha*y
            alpha_list.append(alpha)
        # loop-2: traversal.
        if len(s_list) == 0:
            gamma = 1.
        else: 
            gamma = np.dot(s_list[-1].T, y_list[-1]) / np.dot(y_list[-1].T, y_list[-1])
        r = gamma * q

        for i in range(len(s_list)):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / np.dot(y.T, s)       
            beta = rho * np.dot(y.T, r)
            r += (alpha_list[-(i+1)] - beta) * s

        return r

    # executive function.
    def four_dims_var_optimizer(self,xb,B,y,R,H,h,max_iter,tol):
        '''
        iteration for it. main procedure
        Input
            xb: background state [ndim,]
            B:  background covariance martrix [ndim,ndim]
            y:  real obs.                     [nobs,nstep]
            R:  obs covariance,               [nobs,nstep]
            H:  obs operator.                 [nobs,nstep]
            h:  obs function.
          tol:  tolerance of gradient
        Output
            xa: analysis value.               [ndim,]
        '''
        # dimensions
        K = xb.shape[0]   # state dim
        m = y.shape[0]    # obs dim
        n = y.shape[1]    # obs number

        # invert R and B.
        R_inv = np.linalg.inv(R); B_inv = np.linalg.inv(B)

        # iteration. (x0 newest; xb the old one)
        x_old = xb
        s_list = []
        y_list = []

        grad_old = self.gradient(x_old, xb, B_inv, y, R_inv, H, h, n)
        cost_old = self.cost_function(x_old, xb, B_inv, R_inv, y, h, n)

        for iterate in range(max_iter):
        
            #Limited BFGS
            d = self.LBFGS_update(grad_old,s_list,y_list)

            # line search
            alpha = 1.0
            x_new = x_old + alpha*d                          
            cost_new = self.cost_function(x_new, xb, B_inv, R_inv, y, h, n)

            while (cost_new > cost_old+1e-4*alpha*np.dot(grad_old.T,d)) & (alpha > 0.1):
                alpha *= 0.5
                x_new = x_old + alpha*d                  
                cost_new = self.cost_function(x_new, xb, B_inv, R_inv, y, h, n)

            grad_new = self.gradient(x_new, xb, B_inv, y, R_inv, H, h, n)

            s_list.append(x_new-x_old)
            y_list.append(grad_new-grad_old)

            if len(s_list) > 10:
                s_list.pop(0)
                y_list.pop(0)

            # fourth step: give values.
            x_old = x_new
            grad_old = grad_new
            cost_old = cost_new

            # judgement
            if np.linalg.norm(grad_old) < tol:
                break

        print(f'total times: {iterate+1}')
        print(f'grads norm: {np.linalg.norm(grad_old)}')

        return x_old


class FourDVar_practical:
    """Strong-constraint 4D-Var with sparse observation times."""

    def __init__(self, model, model_ADM, rk4, rk4_adm):
        self.model = model
        self.model_ADM = model_ADM
        self.RK4 = rk4
        self.rk4_adm = rk4_adm

    def _check_inputs(self, xb, B, y, R, idx, N):
        idx = np.asarray(idx, dtype=int)
        if idx.ndim != 1:
            raise ValueError("idx must be a 1-D array of model-step observation indices")
        if len(idx) != y.shape[1]:
            raise ValueError(f"len(idx)={len(idx)} but y has {y.shape[1]} observation times")
        if len(idx) and (idx.min() < 0 or idx.max() >= N):
            raise ValueError(f"observation indices {idx} must lie in [0, {N-1}]")
        if B.shape != (xb.size, xb.size):
            raise ValueError("B has incompatible shape")
        if R.shape != (y.shape[0], y.shape[0]):
            raise ValueError("R has incompatible shape")
        return idx

    def cost_function(self, x, xb, B, R_inv, idx, y, h, N, return_terms=False):
        from scipy.linalg import solve
        K = len(x)
        idx = np.asarray(idx, dtype=int)

        x_traj = np.zeros((K, N), dtype=float)
        x_traj[:, 0] = x
        for i in range(N - 1):
            x_traj[:, i + 1] = self.RK4(self.model, x_traj[:, i])

        dx = x - xb
        Jb = float(dx.T @ solve(B, dx, assume_a='pos'))

        Jo = 0.0
        m = y.shape[0]
        for j, step in enumerate(idx):
            innovation = y[:, j] - h(x_traj[:, step], m)
            Jo += float(innovation.T @ R_inv @ innovation)

        if return_terms:
            return Jb + Jo, Jb, Jo
        return Jb + Jo

    def gradient(self, x, xb, B, y, R_inv, idx, H, h, N):
        from scipy.linalg import solve
        K = len(x)
        m = y.shape[0]
        idx = np.asarray(idx, dtype=int)
        obs_map = {int(step): j for j, step in enumerate(idx)}

        x_traj = np.zeros((K, N), dtype=float)
        x_traj[:, 0] = x
        for i in range(N - 1):
            x_traj[:, i + 1] = self.RK4(self.model, x_traj[:, i])

        x_adj = np.zeros(K, dtype=float)
        for i in range(N - 1, -1, -1):
            if i in obs_map:
                j = obs_map[i]
                innovation = y[:, j] - h(x_traj[:, i], m)
                x_adj += H(x_traj[:, i], m).T @ R_inv @ innovation
            if i > 0:
                x_adj = self.rk4_adm(
                    self.model_ADM, self.model, x_traj[:, i - 1], x_adj
                )

        grad_Jo = -2.0 * x_adj
        grad_Jb = 2.0 * solve(B, x - xb, assume_a='pos')
        return grad_Jb + grad_Jo

    def LBFGS_update(self, grad, s_list, y_list):
        q = -grad.copy()
        alpha_list = []
        rho_list = []

        for s, yv in zip(reversed(s_list), reversed(y_list)):
            sy = float(np.dot(s, yv))
            if sy <= 1e-14:
                continue
            rho = 1.0 / sy
            alpha = rho * np.dot(s, q)
            q -= alpha * yv
            alpha_list.append(alpha)
            rho_list.append(rho)

        if len(s_list) == 0:
            gamma = 1.0
        else:
            s = s_list[-1]
            yv = y_list[-1]
            yy = float(np.dot(yv, yv))
            sy = float(np.dot(s, yv))
            gamma = sy / yy if (sy > 1e-14 and yy > 1e-14) else 1.0

        r = gamma * q

        valid_pairs = [(s, yv) for s, yv in zip(s_list, y_list) if np.dot(s, yv) > 1e-14]
        for k, (s, yv) in enumerate(valid_pairs):
            rho = 1.0 / np.dot(s, yv)
            beta = rho * np.dot(yv, r)
            alpha = alpha_list[-(k + 1)]
            r += (alpha - beta) * s
        return r

    def four_dims_var_optimizer(
        self, xb, B, y, R, idx, n, H, h, max_iter=200, tol=1e-7,
        verbose=True, return_history=False
    ):
        idx = self._check_inputs(xb, B, y, R, idx, n)
        R_inv = np.linalg.inv(R)

        x_old = xb.copy()
        s_list, y_list = [], []
        grad_old = self.gradient(x_old, xb, B, y, R_inv, idx, H, h, n)
        cost_old = self.cost_function(x_old, xb, B, R_inv, idx, y, h, n)

        history = {"cost": [float(cost_old)], "grad_norm": [float(np.linalg.norm(grad_old))], "alpha": []}
        c1 = 1e-4
        alpha_min = 1e-10
        memory = 10

        for iterate in range(max_iter):
            if np.linalg.norm(grad_old) < tol:
                break

            d = self.LBFGS_update(grad_old, s_list, y_list)
            gtd = float(np.dot(grad_old, d))

            # Safeguard: L-BFGS must give a descent direction.
            if (not np.all(np.isfinite(d))) or gtd >= 0.0:
                s_list.clear()
                y_list.clear()
                d = -grad_old
                gtd = -float(np.dot(grad_old, grad_old))

            alpha = 1.0
            accepted = False
            while alpha >= alpha_min:
                x_new = x_old + alpha * d
                cost_new = self.cost_function(x_new, xb, B, R_inv, idx, y, h, n)
                if np.isfinite(cost_new) and cost_new <= cost_old + c1 * alpha * gtd:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                if verbose:
                    print(f"line search failed at iteration {iterate}")
                break

            grad_new = self.gradient(x_new, xb, B, y, R_inv, idx, H, h, n)
            s = x_new - x_old
            yv = grad_new - grad_old
            sy = float(np.dot(s, yv))
            # Keep only curvature pairs that preserve positive-definite inverse-Hessian approximation.
            if sy > 1e-12 * max(1.0, np.linalg.norm(s) * np.linalg.norm(yv)):
                s_list.append(s)
                y_list.append(yv)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)

            x_old, grad_old, cost_old = x_new, grad_new, cost_new
            history["cost"].append(float(cost_old))
            history["grad_norm"].append(float(np.linalg.norm(grad_old)))
            history["alpha"].append(float(alpha))

        if verbose:
            J, Jb, Jo = self.cost_function(x_old, xb, B, R_inv, idx, y, h, n, return_terms=True)
            print(f"total iterations: {len(history['alpha'])}")
            print(f"gradient norm: {np.linalg.norm(grad_old):.6e}")
            print(f"initial cost: {history['cost'][0]:.6e}")
            print(f"final cost:   {J:.6e}  (Jb={Jb:.6e}, Jo={Jo:.6e})")

        if return_history:
            return x_old, history
        return x_old

    def four_dims_var_optimizer_scipy(
        self, xb, B, y, R, idx, n, H, h, max_iter=200, tol=1e-7,
        verbose=True, return_result=False
    ):
        """Robust L-BFGS-B optimizer using the analytic adjoint gradient."""
        from scipy.optimize import minimize

        idx = self._check_inputs(xb, B, y, R, idx, n)
        R_inv = np.linalg.inv(R)

        def fun(x):
            return self.cost_function(x, xb, B, R_inv, idx, y, h, n)

        def jac(x):
            return self.gradient(x, xb, B, y, R_inv, idx, H, h, n)

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
            J, Jb, Jo = self.cost_function(result.x, xb, B, R_inv, idx, y, h, n, return_terms=True)
            print(f"success: {result.success} ({result.message})")
            print(f"iterations: {result.nit}")
            print(f"gradient norm: {np.linalg.norm(result.jac):.6e}")
            print(f"initial cost: {fun(xb):.6e}")
            print(f"final cost:   {J:.6e}  (Jb={Jb:.6e}, Jo={Jo:.6e})")

        if return_result:
            return result.x, result
        return result.x

# 3DVar


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
    ndim, nens = xb.shape; nobs = len(y)
    # step2: get centralized matrix
    IN = np.eye(nens); I = np.ones(nens)
    PI = (IN - np.outer(I,I)/nens)/np.sqrt(nens-1)
    # step3: disturb measurement
    E  = np.random.multivariate_normal(np.zeros(nens), R, size=nobs).T #[nobs, nens]
    D  = y[:,None]@I[None,:] + np.sqrt(nens-1)*E
    # step4: project prior to measurement
    y_model  = ObsOp(xb); Y = y_model@PI
    # step5 : update
    if loc == None:
        W  = loc@Y.T@inv(Y@Y.T+E@E.T)@(D - y_model)
    else:
        W  = loc@Y.T@inv(Y@Y.T+E@E.T)@(D - y_model)
    ## inflation
    xm = xb.mean(aixs=1)
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
    ym = H(xbb); yp = H(xbp)
    # update
    for i in range(D):
        #solve RC = yb get R-1yp transpose: yp.T@R-1
        C = solve(R, yp, assume_a='pos'); CT = C.T
        A_mat = (nens-1)/gamma* np.eye(nens) + loc[i:i,:] @ CT @ yp
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
        w = Pa @ CT @ (y.reshape(-1, 1) - ym)
        if np.iscomplexobj(W):
            W = np.real(W)
        # update locally
        E[i, :] = xbb[i, 0] + xbp[i, :] @ (w + W)

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
    ym = H(xbb); yp = H(xbp)
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
    w = Pa @ CT @ (y.reshape(-1, 1) - ym)
    if np.iscomplexobj(W):
        W = np.real(W)
    # udate
    E = xbb+xbp@(w+W)
    return E

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