import numpy as np
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
    return w

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

def lanczos_iteration(m,n,A,tol,nsv):
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
        r0 = np.full(m,1)
        # calculate matrix-vector dot A_p @ q
        w = A.dot(Q[:,i])
        # Lanczos normalization
        # update 
        if i > 0:
            w = w - beta[i-1] * Q[:,i-1]
        # Orthogonal coefficient
        alpha[i] = np.dot(Q[:,i], w)
        # update
        w = w - alpha[i]*Q[:,i]
        if i > 0:
            w = re_orthogonalize(w,Q[:,:i+1],i)
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
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        #T = build_T(alpha[:i+1],beta[:i])

    return Q, T

np.random.seed(42)

n = 100
B = np.random.randn(n, n)
A = (B + B.T) / 2

P = np.eye(n)

Q, T = lanczos_iteration(n,100,A,1e-7,20)
theta, S = np.linalg.eigh(T)

# Lanczos Ritz vectors
V = Q @ S

# exact eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(A)


# =====================================
# 1. eigenvalue error
# =====================================

eig_error = np.abs(theta - eigenvalues)

print("===== Eigenvalue =====")
print("最大误差:", np.max(eig_error))
print("平均误差:", np.mean(eig_error))


# =====================================
# 2. Ritz residual
# =====================================

residuals = np.zeros(len(theta))

for i in range(len(theta)):
    residuals[i] = np.linalg.norm(
        A @ V[:, i]
        - theta[i] * V[:, i]
    )

print("\n===== Residual =====")
print("最大 residual:", np.max(residuals))
print("平均 residual:", np.mean(residuals))


# =====================================
# 3. Eigenvector similarity
# =====================================

similarities = np.abs(
    np.sum(V * eigenvectors, axis=0)
)

print("\n===== Eigenvector =====")
print("最小 similarity:", np.min(similarities))
print("平均 similarity:", np.mean(similarities))


# =====================================
# 4. Q orthogonality
# =====================================

print("\n===== Q =====")
print(
    "Q orthogonality error:",
    np.linalg.norm(
        Q.T @ Q - np.eye(Q.shape[1])
    )
)