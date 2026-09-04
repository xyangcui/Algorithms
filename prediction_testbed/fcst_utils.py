import numpy as np
from toy_models import RK4, L96
#------------------------------------------------------------------------------
# global function
#------------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
def GSR_process(x):
    from numpy.linalg import qr
    # pass a 2D martrix[n,m] n vectors with m dimensions
    x = x.T
    m = len(x[:,0]); n = len(x[0,:])
    for i in range(1,n):
        col1 = x[:,i]
        col2 = x[:,:i]
        for j in range(i):
            x[:,i] -= np.dot(col1,col2[:,j])/np.dot(col1,col1) * col1

    return x.T

# forecast initialization
# breeding vectors
def breeding_vectors(x2,B,N,M_update,rescaled_dt=0.2,breeding_length=2):
    '''
      breeding vectors.
      Input
        x2: inital state
         B: background covar
         N: ensemble
      Output
        delta: perturbation
    '''

    K  = len(x2)
    x1 = np.zeros([N,K])     # perturb
    for i in range(N):
        x1[i,:] = x2 + np.random.multivariate_normal(np.zeros(K), B)

    initial_delta = x1 - x2
    initial_rms = np.linalg.norm(initial_delta,axis=1)
    #step2: breeding
    breeding_step = int(breeding_length/rescaled_dt)

    for k in range(breeding_step):
        #N ens
        for i in range(N):
            x1[i,:] = M_update(x1[i,:],rescaled_dt) #RK4(L96,x1[i,:],rescaled_dt,F)
        #true value.
        x2 = M_update(x2,rescaled_dt) #RK4(L96,x2,rescaled_dt,F)
        # x1 minus x2
        delta = x1 - x2
        # scaled and add
        delta_rms = np.linalg.norm(delta)

        # rescaled it.
        delta_rms = np.linalg.norm(delta,axis=1)
        masked = delta_rms > 1e-10
        delta[masked,:] *= initial_rms[masked,np.newaxis]/delta_rms[masked,np.newaxis]
        x1 = x2[np.newaxis,:] + delta

    return delta

# singular vectors (theoretical, needs jacobi of TLM, not practical).
def re_orthogonalize(w,Q_sub,k):
    '''re-orthognalize vector w.'''
    # if the first step, return itself.
    if k != 0:
        # first orthogonalize
        alpha_base = Q_sub.T.dot(w)
        w = w - Q_sub.dot(alpha_base)
        # second orthogonalize
        alpha_base = Q_sub.T.dot(w)
        w = w - Q_sub.dot(alpha_base)

def propagator(x, P, r0, CF, TLM, ADM):
    '''use TLM and ADM calculate L.T@L@dx'''
    from scipy.sparse.linalg import spsolve
    # forward integration to get L @ vt
    v  = x/r0
    x1 = TLM(v)
    # use CF to normalize x1. like CF @ x1
    rF = CF(P.dot(x1))
    x2 = np.diag(rF*rF).dot(x1)
    # backward integration to get L.T @ x2
    u  = ADM(x2)
    # u = Norm @ u
    return r0*u

def build_T(alpha, beta):
    alpha = np.asarray(alpha).flatten()
    beta = np.asarray(beta).flatten()
    
    n = len(alpha)
    
    if n == 0:
        return np.array([])
    T = np.diag(alpha)

    if n > 1:
        if len(beta) >= n - 1:
            beta_used = beta[:n-1]
        else:
            beta_used = np.pad(beta, (0, n-1-len(beta)))
        
        T += np.diag(beta_used, 1)
        T += np.diag(beta_used, -1)
    return T

def lanczos_iteration(m,n,P,C0,CF,TLM,ADM,tol,nsv):
    '''
    lanczos iteration to get singular vectors and singular value.
    The problem is A @ x = c* C0 @ x.
    Input
      m: space.
      n: n*nsv.
      P: projection matrix.
      C0: function to get initial norm vector.
      CF: function to get final norm vector.
      TLM: tangent linear model
      ADM: adjoint model
      tol: tolerance
      nsv: the number of singular vectors
    Output
      Q: a set of projection vectors
      T: a matrix in Krylov subspace
    '''
    # 1. initialize (store norm space)
    q = np.random.randn(m)
    q = q / np.linalg.norm(q)
    Q = np.zeros([m,n]); Q[:,0] = q
    T = np.zeros([n,n])
    alpha = np.zeros(n)
    beta = np.zeros(n-1)
    # 2. iteration
    for i in range(n):
        # get initial norm matrix
        r0 = np.sqrt(C0(Q[:,i]))
        r0 = np.full(m,1)
        # calculate matrix-vector dot A_p @ q
        w = propagator(Q[:,i],P,r0,CF,TLM,ADM)
        # Lanczos normalization
        # update 
        if i > 0:
            w = w - beta[i-1] * Q[:,i-1]
        # Orthogonal coefficient
        alpha[i] = np.dot(Q[:,i], w)
        # update
        w = w - alpha[i]*Q[:,i]
        if i > 0:
            re_orthogonalize(w,Q[:,:i],i)
        print(np.dot(w,Q[:,i]))
        # update
        if i < n-1:
            beta[i] = np.linalg.norm(w)
            if beta[i] < tol and i > nsv:
                T = T[:i,:i]
                Q = Q[:,:i]
                break
            else:
                # get new perturbation in physical space.
                Q[:, i+1] = w / beta[i]
                
        # store alpha, beta to T
        #T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        T = build_T(alpha[:i+1],beta[:i])

    return Q, T

def gaussian_sampling(SV, Pa, gamma, nmember, nsv):
    '''sampling parameters to linearly combine SVs'''
    from scipy.stats import truncnorm
    # 1. standardize
    SV_std = SV / Pa
    # 2. norm
    sv_norm = np.linalg.norm(SV_std,axis=0,keepdims=False)
    beta = gamma / sv_norm.mean()
    # 3. sampling [n,nsv]
    return truncnorm.rvs(-3,3,loc=0.,scale=beta,size=(nmember,nsv))

def singular_vectors_theoretical(x2,nsv,M_update,M_TLM,sv_dt,sv_length,nmember,Pa,rescale):
    '''
      singular vectors.
      Input
        x2: inital state
         N: retain numbers the number of members
         M_update: model update
         M_TLM:   Tangent linear operator
         sv_length: sv times
         sv_dt:  sv delta_t
         nmember: the number of member
         Pa: analyze error variance vector
         rescale: an emperical parameter to rescale for more precise ensemble spread.
      Output
        x1: perturbation
    '''
    from numpy.linalg import svd
    import numpy as np

    K  = len(x2)
    # TLM jacobi
    TLM= np.eye(K)
    for i in range(sv_length):
        TLM = M_TLM(x2,sv_dt)
        x2  = M_update(x2,sv_dt)
    # adjoint jacobi
    ADM = TLM.T      
    # SVD analysis
    _,_,Vh = svd(ADM@TLM,full_matrices=True)
    SV = Vh[:nsv,:].T
    # sampling
    ## use analyze error covariance to decide parameters
    Alpha = gaussian_sampling(SV,Pa,rescale,nmember,nsv)       

    return Alpha@SV.T

def singular_vectors(m,nsv,scale,tol,P,C0,CF,TLM,ADM,nmember,Pa,rescale):
    '''
    use lanczos method to get nsv singular vectors and get a ensemble.
    Input
      m: length of space.
      nsv: the number of singular vectors
      scale: determine the size of Krylov subspace
      tol: determine whether to cut iteration
      C0: function to normalize the initial state.
      P[m,m]: project matrix (where to use)
      CF: function to normalize the final state. (perhaps total energy metrics.)
      TLM: function to integrate TLM. (only needs input as self-variable)
      ADM: function to integrate ADM, (like TLM)
      nmember: the number of member
      Pa: analyze error variance vector
      rescale: an emperical parameter to rescale for more precise ensemble spread.
    Output
      ensemble[nmember,m]: a ensemble of forecast members
    '''
    import numpy as np
    n  = int(nsv * scale)
    # 1. lanczos iteration (project to Krylov subspace)
    Q, T = lanczos_iteration(m,n,P,C0,CF,TLM,ADM,tol,nsv)
    print("T 不对称程度:", np.max(np.abs(T - T.T)))
    print("Q 正交性误差:", np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])))
    # 2. SVD the small matrix T
    eigenvalues, eigenvectors = np.linalg.eig(T) 
    # 3. get Ritz vectors
    SV = np.matmul(Q, np.sqrt(eigenvalues)*eigenvectors)[:,:nsv]
    # 4. generate members
    ## use analyze error covariance to decide parameters
    Alpha = gaussian_sampling(SV,Pa,rescale,nmember,nsv)

    return Alpha@SV.T


# type-3: Nonlinear Lyapunov Vectors (NLLVs)
def NLL_vectors(x2,B,N1,N2,M_update,rescaled_dt=0.2,breeding_length=2):
    '''
      NLLVs.
      Input
        x2: inital state
         B: background covar error
         N1: members for select the first N2 vectors.
         N2: output ensemble members
         M_update: update model. params: x and dt
      Output
        delta: perturbation
    '''

    K  = len(x2)
    x1 = np.zeros([N1,K])     # perturb
    for i in range(N1):
        x1[i,:] = x2 + np.random.multivariate_normal(np.zeros(K), B)

    initial_delta = x1 - x2
    initial_rms = np.linalg.norm(initial_delta,axis=1)

    breeding_step = int(breeding_length/rescaled_dt)

    rms_breeding = np.zeros((breeding_step+1,N1))  #[step, ensemble]
    rms_breeding[0,:] = initial_rms
    for k in range(breeding_step):
        for i in range(N1):
            x1[i,:] = M_update(x1[i,:],rescaled_dt)
        #true value.
        x2 = RK4(L96,x2,rescaled_dt)
        # x1 minus x2
        delta = x1 - x2[np.newaxis,:]
        ## related to NLLE
        # calc rms and store.
        delta_rms = np.linalg.norm(delta,axis=1)
        masked = (np.isnan(delta_rms)) | (delta_rms<1e-20)
        delta_rms[masked] += 1e-10

        rms_breeding[k+1,:] = delta_rms

        # calc each member's growth rate and return its rank.
        growth_rate = 1/rescaled_dt*np.log(rms_breeding[k+1,:]/rms_breeding[k,:])
        sorted_indice = np.argsort(growth_rate)[::-1]
        # sort delta and rms as previous indice.
        initial_rms = initial_rms[sorted_indice]
        initial_delta = initial_delta[sorted_indice,:]
        delta = delta[sorted_indice,:]
        rms_breeding = rms_breeding[:,sorted_indice]
        # Gram-Schmidt process
        delta = GSR_process(delta)

        # rescaled it.
        delta_rms = np.linalg.norm(delta,axis=1)
        delta *= initial_rms[:,np.newaxis]/delta_rms[:,np.newaxis]
        x1 = x2[np.newaxis,:] + delta

        # store new rms.
        rms_breeding[k+1,:] = delta_rms    

    return delta[:N2,:]

## second-order exact sampling (SOES)
def construct_constrained_matrix(n):
    '''Householder transformations'''
    from math import sqrt
    from scipy.stats import ortho_group
    # define a vector
    u = np.full(n,1/sqrt(n))
    # project a base vector to direction u.
    v = np.zeros(n); v[0] = 1
    v = v - u
    # get householder matrix
    H = np.eye(n) - 2*(v@v.T)/(v.T@v)
    # random rotation
    R = ortho_group.rvs(dim=n-1)
    # rotate the matrix H
    return H[:,:n-1]@R

def SOES_vectors(base, nmember):
    '''
    Its perturbations precisely keep the first two moments: mean and variance.
    Input
      base[ndim, nsample]: historical records of forecast. analysis error
      nmember: the number of forecast member.
    Ouput
      Omega[nmember, ndim]: a ensemble of n members.
    '''
    from numpy.linalg import svd
    from math import sqrt
    # 1. calculate anomaly
    # 2. SVD anomalous field to get direction and singular values.
    U, Sigma, _ = svd(base)
    # 3. key: construct a constrained random orthogonal matrix. (fullfill the two-order condition)
    Omega = construct_constrained_matrix(nmember) # [nmember, nmember-1]
    # 4. cut off U and Sigma to nmember and restore with Omega.
    Uc = U[:,:nmember-1]; Sigmac = Sigma[:nmember-1,:nmember-1]
    Omega = Uc@Sigmac@Omega.T

    return sqrt(nmember-1)*Omega
