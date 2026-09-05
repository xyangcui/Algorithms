import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm, L96_tlm
from solve_ode import  runge_kuta4, rk4_nl_adm, rk4_nl_tlm
from fcst_utils import singular_vectors
import numpy as np
import pickle

# ============================================================
#  SV / Lanczos debug script
#
#  目标：
#    1. 用 Lanczos 计算 A = L^T P^T P L 的主特征向量
#    2. 检查 Q 的正交性
#    3. 检查 Lanczos 构造的 T 是否等于 Q^T A Q
#    4. 检查 A 是否对称、半正定
#    5. 检查 TLM / ADM 是否满足伴随关系
#
#  你主要需要修改：
#    - 下方 “MODEL INTERFACE” 部分
#    - main() 中的数据读取 / initial_state
# ============================================================


# ============================================================
# 1. Linear algebra helpers
# ============================================================

def orthogonality_error(Q):
    """Return ||Q^T Q - I||_F."""
    if Q.size == 0:
        return 0.0
    k = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(k))


def max_orthogonality_error(Q):
    """Return max |Q^T Q - I|."""
    if Q.size == 0:
        return 0.0
    k = Q.shape[1]
    return np.max(np.abs(Q.T @ Q - np.eye(k)))


def reorthogonalize(w, Q, passes=2):
    """
    Full reorthogonalization.

    Parameters
    ----------
    w : (m,) ndarray
    Q : (m,k) ndarray
        Existing orthonormal basis.
    passes : int
        2 is usually robust enough.
    """
    if Q.size == 0 or Q.shape[1] == 0:
        return w

    for _ in range(passes):
        coeff = Q.T @ w
        w = w - Q @ coeff

    return w


# ============================================================
# 2. MODEL INTERFACE
#
# Replace / adapt this section to your project.
# ============================================================

def make_tlm_operator(TLM_func, z0, nstep, dt):
    """
    Wrap your TLM implementation into a one-argument linear operator.

    Expected original interface:
        TLM_func(dx, z0, nstep, dt) -> propagated perturbation
    """
    def op(dx):
        return TLM_func(dx, z0, nstep, dt)
    return op


def make_adm_operator(ADM_func, z0, nstep, dt):
    """
    Wrap your ADM implementation into a one-argument adjoint operator.

    Expected original interface:
        ADM_func(dy, z0, nstep, dt) -> backward propagated adjoint vector
    """
    def op(dy):
        return ADM_func(dy, z0, nstep, dt)
    return op


# ============================================================
# 3. Adjoint test
# ============================================================

def test_adjoint(TLM, ADM, m, num_tests=5, tol=1e-8, seed=0):
    """
    Test:
        <L x, y> == <x, L^T y>

    This should be tested BEFORE debugging Lanczos.
    """
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 70)
    print("ADJOINT TEST")
    print("=" * 70)

    max_rel = 0.0

    for i in range(num_tests):
        x = rng.standard_normal(m)
        y = rng.standard_normal(m)

        Lx = TLM(x)
        LTy = ADM(y)

        lhs = np.dot(Lx, y)
        rhs = np.dot(x, LTy)

        abs_err = abs(lhs - rhs)
        scale = max(abs(lhs), abs(rhs), 1.0)
        rel_err = abs_err / scale

        max_rel = max(max_rel, rel_err)

        print(
            f"test {i+1:2d}: "
            f"<Lx,y>={lhs:+.12e}, "
            f"<x,L^Ty>={rhs:+.12e}, "
            f"abs={abs_err:.3e}, "
            f"rel={rel_err:.3e}"
        )

    print(f"max relative adjoint error = {max_rel:.3e}")

    if max_rel > tol:
        print("WARNING: adjoint test FAILED.")
        print("Lanczos for L^T L may not behave as a symmetric problem.")
    else:
        print("Adjoint test PASSED.")

    return max_rel


# ============================================================
# 4. Define the symmetric positive-semidefinite operator
#
#        A = L^T P^T P L
#
# For P = I:
#
#        A = L^T L
#
# IMPORTANT:
# Do NOT put q-dependent nonlinear metric operations here.
# Lanczos requires A to be a FIXED LINEAR operator.
# ============================================================

def make_A_operator(TLM, ADM, P=None):
    """
    Construct A(x) = L^T P^T P L x.

    P can be:
      - None: identity
      - ndarray shape (r,m)
    """
    if P is None:
        def A(x):
            return ADM(TLM(x))
    else:
        P = np.asarray(P)

        def A(x):
            y = TLM(x)
            y = P @ y
            y = P.T @ y
            return ADM(y)

    return A


# ============================================================
# 5. Basic operator diagnostics
# ============================================================

def test_operator_symmetry(A, m, num_tests=5, seed=1):
    """
    Test:
        <x, A y> == <A x, y>
    """
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 70)
    print("OPERATOR SYMMETRY TEST")
    print("=" * 70)

    errs = []

    for i in range(num_tests):
        x = rng.standard_normal(m)
        y = rng.standard_normal(m)

        Ax = A(x)
        Ay = A(y)

        lhs = np.dot(x, Ay)
        rhs = np.dot(Ax, y)

        err = abs(lhs - rhs)
        scale = max(abs(lhs), abs(rhs), 1.0)
        rel = err / scale
        errs.append(rel)

        print(
            f"test {i+1:2d}: "
            f"<x,Ay>={lhs:+.12e}, "
            f"<Ax,y>={rhs:+.12e}, "
            f"rel={rel:.3e}"
        )

    print(f"max relative symmetry error = {max(errs):.3e}")
    return max(errs)


def test_operator_psd(A, m, num_tests=10, seed=2):
    """
    Test random Rayleigh quotients x^T A x.

    For A = L^T P^T P L, these should be >= 0
    up to roundoff.
    """
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 70)
    print("POSITIVE-SEMIDEFINITE TEST")
    print("=" * 70)

    vals = []

    for i in range(num_tests):
        x = rng.standard_normal(m)
        x /= np.linalg.norm(x)

        Ax = A(x)
        rq = np.dot(x, Ax)
        vals.append(rq)

        print(f"test {i+1:2d}: x^T A x = {rq:+.12e}")

    vals = np.asarray(vals)
    print(f"minimum sampled Rayleigh quotient = {vals.min():+.12e}")

    if vals.min() < -1e-10:
        print("WARNING: A does not look positive semidefinite.")
    else:
        print("PSD random test looks OK.")

    return vals


# ============================================================
# 6. Lanczos iteration
# ============================================================

def lanczos(
    A,
    m,
    niter,
    tol=1e-12,
    reorth=True,
    reorth_passes=2,
    seed=3,
    verbose=True,
):
    """
    Symmetric Lanczos iteration.

    Parameters
    ----------
    A : callable
        Linear symmetric operator.
    m : int
        Physical-space dimension.
    niter : int
        Krylov dimension. Automatically clipped to m.
    tol : float
        Breakdown tolerance.
    reorth : bool
        Full reorthogonalization.
    reorth_passes : int
        Number of full reorthogonalization passes.
    """

    # Cannot have more than m orthonormal columns in R^m
    niter = min(int(niter), int(m))

    rng = np.random.default_rng(seed)

    Q = np.zeros((m, niter), dtype=float)
    alpha = np.zeros(niter, dtype=float)
    beta = np.zeros(max(niter - 1, 0), dtype=float)

    q = rng.standard_normal(m)
    q /= np.linalg.norm(q)
    Q[:, 0] = q

    k_eff = niter

    if verbose:
        print("\n" + "=" * 70)
        print("LANCZOS")
        print("=" * 70)
        print(f"physical dimension m = {m}")
        print(f"requested Krylov dimension = {niter}")

    for i in range(niter):
        q_i = Q[:, i]

        # w = A q_i
        w = A(q_i)

        # Remove previous Lanczos direction
        if i > 0:
            w = w - beta[i - 1] * Q[:, i - 1]

        # Diagonal coefficient
        alpha[i] = np.dot(q_i, w)

        # Remove current direction
        w = w - alpha[i] * q_i

        # Numerical stabilization
        if reorth:
            w = reorthogonalize(
                w,
                Q[:, : i + 1],
                passes=reorth_passes,
            )

        # Last requested vector: no need for beta
        if i == niter - 1:
            k_eff = niter
            break

        beta_i = np.linalg.norm(w)
        beta[i] = beta_i

        if verbose:
            err = orthogonality_error(Q[:, : i + 1])
            maxerr = max_orthogonality_error(Q[:, : i + 1])

            print(
                f"iter={i:3d}  "
                f"alpha={alpha[i]:+.6e}  "
                f"beta={beta_i:.6e}  "
                f"orth_F={err:.3e}  "
                f"orth_max={maxerr:.3e}"
            )

        # Exact / numerical Krylov breakdown
        if beta_i < tol:
            k_eff = i + 1
            if verbose:
                print(
                    f"Lanczos breakdown at iteration {i}: "
                    f"beta={beta_i:.3e}"
                )
            break

        Q[:, i + 1] = w / beta_i

    # Truncate to actually generated basis
    Q = Q[:, :k_eff]
    alpha = alpha[:k_eff]

    # Construct symmetric tridiagonal T
    T = np.diag(alpha)

    if k_eff > 1:
        b = beta[: k_eff - 1]
        T += np.diag(b, 1)
        T += np.diag(b, -1)

    return Q, T, alpha, beta[: max(k_eff - 1, 0)]


# ============================================================
# 7. Explicit projected matrix:
#
#        T_exact = Q^T A Q
#
# This is one of the most important diagnostics.
# ============================================================

def projected_operator(A, Q):
    AQ = np.column_stack([A(Q[:, j]) for j in range(Q.shape[1])])
    T_exact = Q.T @ AQ
    return T_exact, AQ


def diagnose_lanczos(A, Q, T):
    """
    Compare Lanczos tridiagonal T with actual Q^T A Q.
    """
    print("\n" + "=" * 70)
    print("LANCZOS DIAGNOSTICS")
    print("=" * 70)

    k = Q.shape[1]

    q_orth = orthogonality_error(Q)
    q_max = max_orthogonality_error(Q)

    print(f"Q shape                     = {Q.shape}")
    print(f"||Q^TQ-I||_F               = {q_orth:.6e}")
    print(f"max|Q^TQ-I|                = {q_max:.6e}")
    print(f"||T-T^T||_F                = {np.linalg.norm(T-T.T):.6e}")

    T_exact, AQ = projected_operator(A, Q)

    proj_sym = np.linalg.norm(T_exact - T_exact.T)
    diff = np.linalg.norm(T - T_exact)
    rel_diff = diff / max(np.linalg.norm(T_exact), 1e-30)

    print(f"||Q^TAQ-(Q^TAQ)^T||_F      = {proj_sym:.6e}")
    print(f"||T-Q^TAQ||_F              = {diff:.6e}")
    print(f"relative ||T-Q^TAQ||       = {rel_diff:.6e}")

    # Symmetrize explicit projection only for diagnostic eigenspectrum
    T_exact_sym = 0.5 * (T_exact + T_exact.T)

    eig_T = np.linalg.eigvalsh(T)
    eig_exact = np.linalg.eigvalsh(T_exact_sym)

    print("\neig(T):")
    print(eig_T)

    print("\neig(sym(Q^T A Q)):")
    print(eig_exact)

    print(f"\nmin eig(T)                 = {eig_T.min():+.12e}")
    print(f"min eig(sym(Q^TAQ))        = {eig_exact.min():+.12e}")

    print("\nInterpretation:")
    if eig_T.min() < -1e-10 and eig_exact.min() >= -1e-10:
        print(
            "  -> T has a real negative eigenvalue but Q^T A Q does not.\n"
            "     The three-term Lanczos representation is being violated."
        )
    elif eig_exact.min() < -1e-10:
        print(
            "  -> Q^T A Q itself has a significant negative eigenvalue.\n"
            "     Check TLM/ADM consistency and the definition of A."
        )
    else:
        print(
            "  -> No significant negative eigenvalues detected.\n"
            "     Tiny negative values near machine precision are roundoff."
        )

    return {
        "T_exact": T_exact,
        "AQ": AQ,
        "eig_T": eig_T,
        "eig_T_exact": eig_exact,
        "orth_error": q_orth,
        "T_projection_error": diff,
        "T_projection_relative_error": rel_diff,
    }


# ============================================================
# 8. Ritz vectors / singular vectors
# ============================================================

def extract_singular_vectors(Q, T, nsv):
    """
    For A = L^T L:

        T y = lambda y
        SV = Q y
        singular value = sqrt(lambda)

    IMPORTANT:
        sqrt(lambda) should NOT be multiplied into the direction SV.
    """

    eigvals, eigvecs = np.linalg.eigh(T)

    # Largest eigenvalues first
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    nsv = min(int(nsv), Q.shape[1])

    selected_vals = eigvals[:nsv]
    selected_vecs = eigvecs[:, :nsv]

    SV = Q @ selected_vecs

    # For PSD operator, only tiny negative values should exist.
    singular_values = np.sqrt(np.clip(selected_vals, 0.0, None))

    return SV, singular_values, selected_vals


# ============================================================
# 9. Optional dense validation
#
# For small systems such as Lorenz-96 K=40, explicitly forming A
# is extremely useful for debugging.
# ============================================================

def form_dense_matrix(A, m):
    """
    Explicitly form the matrix representation of A by applying A
    to coordinate basis vectors.
    """
    E = np.eye(m)
    A_dense = np.column_stack([A(E[:, j]) for j in range(m)])
    return A_dense


def diagnose_dense_A(A, m):
    print("\n" + "=" * 70)
    print("DENSE A DIAGNOSTICS")
    print("=" * 70)

    A_dense = form_dense_matrix(A, m)

    sym_err = np.linalg.norm(A_dense - A_dense.T)
    rel_sym = sym_err / max(np.linalg.norm(A_dense), 1e-30)

    A_sym = 0.5 * (A_dense + A_dense.T)
    eigvals = np.linalg.eigvalsh(A_sym)

    print(f"||A-A^T||_F                = {sym_err:.6e}")
    print(f"relative symmetry error    = {rel_sym:.6e}")
    print(f"min eig(sym(A))            = {eigvals.min():+.12e}")
    print(f"max eig(sym(A))            = {eigvals.max():+.12e}")

    return A_dense, eigvals


# ============================================================
# 10. Example integration with your project
# ============================================================

def main():
    """
    Adapt this section to your project.

    Below is written to resemble your existing SV.py.

    Uncomment and adjust imports / file paths as needed.
    """

    # --------------------------------------------------------
    # Example project imports
    # --------------------------------------------------------
    #
    #
    K = 40
    F = 8.0
    #
    #
    def TLM_model(x, zt, N, dt):
         for _ in range(N):
            zt, x = rk4_nl_tlm(
                 lambda a, b: L96_tlm(a, b),
                 lambda a: L96(a, F),
                 zt,
                 x,
                 dt,
             )
         return x
    #
    #
    def ADM_model(x, zt, N, dt):
         global zbase
         zbase = np.zeros((K, N + 1), dtype=float)
         zbase[:, 0] = zt
    
         for i in range(N):
             zbase[:, i + 1] = runge_kuta4(
                 lambda a: L96(a, F),
                 zbase[:, i],
                 dt,
             )
    
         for i in range(N, 0, -1):
             x = rk4_nl_adm(
                 lambda a, b: L96_adm(a, b),
                 lambda a: L96(a, F),
                 zbase[:, i - 1],
                 x,
                 dt,
             )
    
         return x
    #
    #
    # # load your initial state
    with open("ensembleDA.pkl", "rb") as f:
         initial_state = pickle.load(f)
    #
    icase = 0
    z0 = initial_state[:, 0, icase]
    #
    sv_t = 1.0
    sv_dt = 0.1
    nstep = int(sv_t / sv_dt)
    #
    TLM = make_tlm_operator(
         TLM_model,
         z0=z0,
         nstep=nstep,
         dt=sv_dt,
     )
    #
    ADM = make_adm_operator(
         ADM_model,
         z0=z0,
         nstep=nstep,
         dt=sv_dt,
     )
    #
    # # Projection / final metric.
    # # First debug with identity.
    P = np.eye(K)
    #
    A = make_A_operator(TLM, ADM, P=P)
    #
    #
    # # ------------------------------------------------------
    # # Step 1: verify TLM / ADM
    # # ------------------------------------------------------
    test_adjoint(
         TLM=TLM,
         ADM=ADM,
         m=K,
         num_tests=5,
         tol=1e-8,
     )
    #
    #
    # # ------------------------------------------------------
    # # Step 2: verify A
    # # ------------------------------------------------------
    test_operator_symmetry(A, K)
    test_operator_psd(A, K)
    #
    # # K=40 is small, so dense validation is recommended.
    A_dense, eigA = diagnose_dense_A(A, K)
    #
    #
    # # ------------------------------------------------------
    # # Step 3: Lanczos
    # # ------------------------------------------------------
    nsv = 10
    scale = 3
    #
    # # Critical:
    # # Krylov dimension cannot exceed physical dimension K.
    niter = min(K, nsv * scale)
    #
    Q, T, alpha, beta = lanczos(
         A=A,
         m=K,
         niter=niter,
         tol=1e-12,
         reorth=True,
         reorth_passes=2,
         seed=3,
         verbose=True,
     )
    #
    #
    # # ------------------------------------------------------
    # # Step 4: compare T with Q^T A Q
    # # ------------------------------------------------------
    result = diagnose_lanczos(A, Q, T)
    #
    #
    # # ------------------------------------------------------
    # # Step 5: extract SVs
    # # ------------------------------------------------------
    SV, sigma, lambdas = extract_singular_vectors(
         Q,
         T,
         nsv=nsv,
     )
    #
    print("\n" + "=" * 70)
    print("SV RESULT")
    print("=" * 70)
    print("SV shape =", SV.shape)
    print("eigenvalues =", lambdas)
    print("singular values =", sigma)
    print(
         "SV orthogonality =",
         np.linalg.norm(
             SV.T @ SV - np.eye(SV.shape[1])
         )
     )

    print(
        "This file is a debug template.\n"
        "Open main() and uncomment/adapt the project-specific section."
    )


if __name__ == "__main__":
    main()
