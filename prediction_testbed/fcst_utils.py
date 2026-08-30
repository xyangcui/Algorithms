import numpy as np

#------------------------------------------------------------------------------
# global function
#------------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
def GSR_process(x):
    from numpy.linalg import qr
    # pass a 2D martrix[m,n] n vectors with m dimensions
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
def singular_vectors_theoretical(x2,N,M_update,M_TLM,sv_dt,sv_length,scale_factor=5.):
    '''
      singular vectors.
      Input
        x2: inital state
         N: retain numbers the number of members
         M_update: model update
         M_TLM:   Tangent linear operator
         sv_length: sv times
         sv_dt:  sv delta_t
         scale: scale factor.
      Output
        x1: perturbation
    '''

    from numpy.linalg import svd
    import numpy as np

    K  = len(x2)
    x1 = np.zeros((N,K))
    # get TLM
    TLM= np.eye(K)
    for i in range(sv_length):
        TLM = M_TLM(x2,sv_dt)
        x2 = M_update(x2,sv_dt)

    ADM = TLM.T      # adjoint model

    _,_,Vh = svd(ADM@TLM,full_matrices=True)
    V = Vh.T
    #scaled.
    norms = np.linalg.norm(V[:,:N], axis=1, keepdims=True)
    norms[norms == 0] = 1
    
    scale_factors = scale_factor / norms #np.random.uniform(0, 1e-5, (1, N)) / norms
    x1 = V[:,:N] * scale_factors    

    return x1.T


def re_orthogonalize(w,Q_sub,k):
    '''re-orthognalize vector w.'''
    # if the first step, return itself.
    if k != 0:
        # first orthogonalize
        alpha_base = Q_sub.T@w
        w = w - Q_sub@alpha_base
        # second orthogonalize
        alpha_base = Q_sub.T@w
        w = w - Q_sub@alpha_base

    return w

def propagator(x, P, C0, C0T, CF, CF_trans, TLM, ADM):
    '''use TLM and ADM calculate L.T@L@dx'''
    # 1. forward integration
    x1 = TLM(C0.dot(x))
    # 2. normalize x1
    xt1 = CF(P@x1)
    xt2 = CF_trans(xt1)
    xt  = P.T@xt2
    # 3. backward integration
    x2 = ADM(xt)
    # 4. normalize x2
    v  = C0T.dot(x2)

    return v

def lanczos_iteration(m,n,P,C0,C0T,CF,CF_trans,TLM,ADM,tol,nsv):
    '''
    lanczos iteration to get singular vectors and singular value.
    Input
      m: space.
      n: n*nsv.
      P: projection matrix.
      C0: initial matrix
      C0T: T C0
      TLM: tangent linear model
      ADM: adjoint model
      tol: tolerance
      nsv: the number of singular vectors
    Output
      Q: a set of projection vectors
      T: a matrix in Krylov subspace
    '''
    # 1. initialize
    raw = np.random.randn(m)
    q = (raw - raw.mean())/raw.std()
    Q = np.zeros([m,n]); Q[:,0] = q
    T = np.zeros([n,n])
    w = q.copy()
    q_new = q; q_old = np.zeros(m)
    beta  = 0.
    # 2. iteration
    for i in range(n):
        # calculate matrix-vector dot
        w = propagator(w,P,C0,C0T,CF,CF_trans,TLM,ADM)
        # Lanczos normalization
        alpha = np.dot(w,q_new)
        w = w - alpha*q_new - beta*q_old
        w = re_orthogonalize(w,Q[:,:i],i)
        beta = np.linalg.norm(w)
        if beta < tol & i > nsv:
            T = T[:i,:i]
            Q = Q[:,:i]
            break
        else:
            q_old = q_new
            q_new = w/beta
        # store alpha, beta to T
        T[i,i] = alpha
        if i < n-1:
            T[i,i+1] = beta; T[i+1,i] = beta
        # store q to Q
        Q[:,i+1] = q_new

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
    return truncnorm(-3,3,loc=0.,scale=beta,size=(nmember,nsv))

def singular_vectors(m,nsv,scale,tol,C0,P,CF,CF_trans,TLM,ADM,nmember,Pa,rescale):
    '''
    use lanczos method to get nsv singular vectors and get a ensemble.
    Input
      m: length of space.
      nsv: the number of singular vectors
      scale: determine the size of Krylov subspace
      tol: determine whether to cut iteration
      C0[m,m]: norm matrix to determine initial state; analyze error.
      P[m,m]: project matrix (where to use)
      CF: function to get norm matrix (determine evolving direction) perhaps total energy metrics.
      TLM: function to integrate TLM. (only needs input as self-variable)
      ADM: function to integrate ADM, (like TLM)
      nmember: the number of member
      Pa: analyze error variance vector
      rescale: an emperical parameter to rescale for more precise ensemble spread.
    Output
      ensemble[nmember,m]: a ensemble of forecast members
    '''
    import numpy as np
    from scipy.sparse import csr_matrix
    n  = int(nsv * scale)
    C0 = csr_matrix(C0); C0T = C0.T
    # 1. lanczos iteration (project to Krylov subspace)
    Q, T = lanczos_iteration(m,n,P,C0,C0T,CF,CF_trans,TLM,ADM,tol,nsv)
    # 2. SVD the small matrix T
    eigenvalues, eigenvectors = np.linalg.eig(T)  
    # 3. get Ritz vectors
    SV = np.matmul(Q, np.sqrt(eigenvalues)*eigenvectors,out=Q)[:,:nsv]
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
        x2 = RK4(L96,x2,rescaled_dt,F)
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
      base[ndim, nsample]: historical records of forecast.
      nmember: the number of forecast member.
    Ouput
      Omega[nmember, ndim]: a ensemble of n members.
    '''
    from numpy.linalg import svd
    from math import sqrt
    # 1. calculate anomaly
    base_anomaly = base - base.mean(aixs=1)
    # 2. SVD anomalous field to get direction and singular values.
    U, Sigma, _ = svd(base_anomaly)
    # 3. key: construct a constrained random orthogonal matrix. (fullfill the two-order condition)
    Omega = construct_constrained_matrix(nmember) # [nmember, nmember-1]
    # 4. cut off U and Sigma to nmember and restore with Omega.
    Uc = U[:,:nmember-1]; Sigmac = Sigma[:nmember-1,:nmember-1]
    Omega = Uc@Sigmac@Omega.T

    return sqrt(nmember-1)*Omega
