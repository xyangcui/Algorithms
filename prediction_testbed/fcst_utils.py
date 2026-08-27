import numpy as np

# forecast initialization
# type-1 breeding vectors
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
    rescaled_dt = 0.2
    breeding_length = 2
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

# type 2: singular vectors
def singular_vectors(x2,N,M_update,M_TLM,sv_dt,sv_length,scale_factor=5.):
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


# type-3: Nonlinear Lyapunov Vectors (NLLVs)
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