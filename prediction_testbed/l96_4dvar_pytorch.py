import os
import pickle
import numpy as np
import torch

# ============================================================
# Configuration
# ============================================================
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = 40
F = 8.0
dt = 0.05
nd = 4


# ============================================================
# Torch observation operator and Lorenz-96 model
# ============================================================
def h_torch(x, obs_dim):
    """Observation function h(x). Supports observing all or evenly spaced state variables."""
    n = x.shape[0]
    m = obs_dim
    if n % m != 0:
        raise ValueError(f"state dimension {n} must be divisible by obs_dim {m}")
    di = n // m
    indices = torch.arange(m, device=x.device) * di + (di - 1)
    return x[indices]


def L96_torch(state, forcing=F):
    """Lorenz-96 RHS, fully differentiable."""
    x = state
    return (
        (torch.roll(x, shifts=-1, dims=0) - torch.roll(x, shifts=2, dims=0))
        * torch.roll(x, shifts=1, dims=0)
        - x
        + forcing
    )


def RK4_torch(rhs, state, dt, *args):
    """Differentiable RK4 step."""
    k1 = rhs(state, *args)
    k2 = rhs(state + 0.5 * dt * k1, *args)
    k3 = rhs(state + 0.5 * dt * k2, *args)
    k4 = rhs(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# NumPy forward model for truth/forecast generation only
# ============================================================
def L96_numpy(state, forcing=F):
    x = state
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def RK4_numpy(rhs, state, dt, *args):
    k1 = rhs(state, *args)
    k2 = rhs(state + 0.5 * dt * k1, *args)
    k3 = rhs(state + 0.5 * dt * k2, *args)
    k4 = rhs(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# 4D-Var cost function
# ============================================================
def cost_function_torch(x0, xb, B_inv, y, R_inv, N):
    """
    Strong-constraint 4D-Var cost:

        J(x0) = (x0-xb)^T B^-1 (x0-xb)
              + sum_i [y_i-h(x_i)]^T R^-1 [y_i-h(x_i)]

    where x_{i+1} is obtained by RK4 integration of Lorenz-96.
    """
    dx = x0 - xb
    Jb = dx @ B_inv @ dx

    state = x0
    Jo = torch.zeros((), dtype=x0.dtype, device=x0.device)

    for i in range(N):
        innovation = y[:, i] - h_torch(state, obs_dim=y.shape[0])
        Jo = Jo + innovation @ R_inv @ innovation

        if i < N - 1:
            state = RK4_torch(L96_torch, state, dt, F)

    return Jb + Jo


# ============================================================
# Pure PyTorch 4D-Var optimizer
# ============================================================
def four_dims_var_optimizer_torch(
    xb,
    B,
    y,
    R,
    max_iter=100,
    lr=0.2,
    history_size=20,
    tolerance_grad=1e-7,
    tolerance_change=1e-10,
    verbose=True,
):
    """
    Optimize the 4D-Var initial state with torch.optim.LBFGS.

    Key point:
      - x0 is the optimization variable.
      - cost_function_torch builds the forward RK4 graph.
      - loss.backward() performs reverse-mode AD (discrete adjoint).
      - torch.optim.LBFGS performs the quasi-Newton update and strong-Wolfe line search.
    """
    xb_t = torch.as_tensor(xb, dtype=DTYPE, device=DEVICE)
    B_t = torch.as_tensor(B, dtype=DTYPE, device=DEVICE)
    y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
    R_t = torch.as_tensor(R, dtype=DTYPE, device=DEVICE)

    # Solve instead of explicit inverse would be possible, but these matrices are small here.
    B_inv_t = torch.inverse(B_t)
    R_inv_t = torch.inverse(R_t)
    N = y_t.shape[1]

    # Control variable / analysis initial condition.
    x0 = xb_t.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x0],
        lr=lr,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
    )

    eval_count = 0
    last_good = {
        "loss": None,
        "grad_norm": None,
        "x": x0.detach().clone(),
    }

    def closure():
        nonlocal eval_count
        optimizer.zero_grad()

        loss = cost_function_torch(x0, xb_t, B_inv_t, y_t, R_inv_t, N)

        # Fail early instead of allowing NaN/Inf to contaminate LBFGS history.
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "4D-Var cost became NaN/Inf. Try a smaller lr, shorter assimilation window, "
                "or check whether the forward L96 trajectory is diverging."
            )

        loss.backward()

        if x0.grad is None or not torch.all(torch.isfinite(x0.grad)):
            raise FloatingPointError(
                "4D-Var gradient became NaN/Inf during autograd/backpropagation."
            )

        grad_norm = torch.norm(x0.grad)
        eval_count += 1

        last_good["loss"] = float(loss.detach().cpu())
        last_good["grad_norm"] = float(grad_norm.detach().cpu())
        last_good["x"] = x0.detach().clone()

        if verbose and (eval_count == 1 or eval_count % 10 == 0):
            xmax = float(torch.max(torch.abs(x0.detach())).cpu())
            print(
                f"eval={eval_count:4d}  "
                f"J={last_good['loss']:.6e}  "
                f"|g|={last_good['grad_norm']:.6e}  "
                f"max|x0|={xmax:.4f}"
            )

        return loss

    try:
        optimizer.step(closure)
    except FloatingPointError as exc:
        # Return the most recent finite state, rather than a corrupted NaN state.
        print(f"LBFGS stopped safely: {exc}")
        with torch.no_grad():
            x0.copy_(last_good["x"])

    # Final diagnostics on a fresh graph.
    optimizer.zero_grad()
    final_loss = cost_function_torch(x0, xb_t, B_inv_t, y_t, R_inv_t, N)
    final_loss.backward()
    final_grad_norm = torch.norm(x0.grad)

    if verbose:
        print(f"device          : {DEVICE}")
        print(f"function evals  : {eval_count}")
        print(f"final cost      : {float(final_loss.detach().cpu()):.6e}")
        print(f"final grad norm : {float(final_grad_norm.detach().cpu()):.6e}")

    return x0.detach().cpu().numpy()


# ============================================================
# Optional AD gradient check
# ============================================================
def gradient_autodiff(x, xb, B_inv, y, R_inv, N):
    x_t = torch.tensor(x, dtype=DTYPE, device=DEVICE, requires_grad=True)
    xb_t = torch.as_tensor(xb, dtype=DTYPE, device=DEVICE)
    B_inv_t = torch.as_tensor(B_inv, dtype=DTYPE, device=DEVICE)
    y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
    R_inv_t = torch.as_tensor(R_inv, dtype=DTYPE, device=DEVICE)

    J = cost_function_torch(x_t, xb_t, B_inv_t, y_t, R_inv_t, N)
    grad, = torch.autograd.grad(J, x_t)
    return grad.detach().cpu().numpy()


def cost_function_numpy_interface(x, xb, B_inv, y, R_inv, N):
    x_t = torch.as_tensor(x, dtype=DTYPE, device=DEVICE)
    xb_t = torch.as_tensor(xb, dtype=DTYPE, device=DEVICE)
    B_inv_t = torch.as_tensor(B_inv, dtype=DTYPE, device=DEVICE)
    y_t = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
    R_inv_t = torch.as_tensor(R_inv, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        J = cost_function_torch(x_t, xb_t, B_inv_t, y_t, R_inv_t, N)
    return float(J.cpu())


def directional_gradient_check(x, xb, B_inv, y, R_inv, N, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=x.shape)
    p /= np.linalg.norm(p)

    g = gradient_autodiff(x, xb, B_inv, y, R_inv, N)
    ad = np.dot(g, p)

    jp = cost_function_numpy_interface(x + eps * p, xb, B_inv, y, R_inv, N)
    jm = cost_function_numpy_interface(x - eps * p, xb, B_inv, y, R_inv, N)
    fd = (jp - jm) / (2.0 * eps)

    rel_error = abs(ad - fd) / max(1.0, abs(ad), abs(fd))
    print(f"AD directional derivative : {ad:.12e}")
    print(f"FD directional derivative : {fd:.12e}")
    print(f"relative error             : {rel_error:.3e}")
    return rel_error


# ============================================================
# OSSE example
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Spinup.
    x0 = np.repeat(F, K).astype(np.float64)
    x0[K // 2] += 0.01
    x0_true = x0.copy()
    for _ in range(int(1000 * nd)):
        x0_true = RK4_numpy(L96_numpy, x0_true, dt, F)

    # A shorter local climatology is enough for this runnable example.
    # Replace with your cached long x_real if desired.
    cache_path = "x_real"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            x_real = pickle.load(f)
    else:
        nt_clim = 5000
        x_real = np.zeros((nt_clim + 1, K), dtype=np.float64)
        x_real[0] = x0_true
        for i in range(nt_clim):
            x_real[i + 1] = RK4_numpy(L96_numpy, x_real[i], dt, F)

    x_std = np.std(x_real, axis=0)
    x_std = np.maximum(x_std, 1e-6)

    # Truth trajectory.
    x_start = x_real[min(1510, len(x_real) - 1)].copy()
    x_truth = np.zeros((10 * nd + 1, K), dtype=np.float64)
    x_truth[0] = x_start
    for j in range(10 * nd):
        x_truth[j + 1] = RK4_numpy(L96_numpy, x_truth[j], dt, F)

    # Observations: IMPORTANT: use copy(), not x_obs = x_truth.
    obs_noise_std = 0.28 * x_std
    x_obs = x_truth.copy()
    x_obs += rng.normal(0.0, obs_noise_std, size=x_obs.shape)

    # Covariances.
    B = np.diag(x_std ** 2)
    R = np.diag(obs_noise_std ** 2)

    # Background initial state.
    x_init = x_truth[0] + rng.normal(0.0, 0.10 * x_std, size=K)

    # No-DA forecast.
    xb_forecast = np.zeros_like(x_truth)
    xb_forecast[0] = x_init
    for j in range(10 * nd):
        xb_forecast[j + 1] = RK4_numpy(L96_numpy, xb_forecast[j], dt, F)

    # Assimilate first 10 observation times, as in your original example.
    y_assim = x_obs[:10].T

    # Optional derivative verification before optimization.
    B_inv = np.linalg.inv(B)
    R_inv = np.linalg.inv(R)
    directional_gradient_check(
        x_init, x_init, B_inv, y_assim, R_inv, y_assim.shape[1]
    )

    # Pure PyTorch L-BFGS 4D-Var.
    xa0 = four_dims_var_optimizer_torch(
        x_init,
        B,
        y_assim,
        R,
        max_iter=100,
        lr=0.2,
        history_size=20,
        verbose=True,
    )

    # Analysis forecast.
    xa = np.zeros_like(x_truth)
    xa[0] = xa0
    for j in range(10 * nd):
        xa[j + 1] = RK4_numpy(L96_numpy, xa[j], dt, F)

    rmse_b = np.sqrt(np.mean((xb_forecast - x_truth) ** 2, axis=1))
    rmse_a = np.sqrt(np.mean((xa - x_truth) ** 2, axis=1))

    print(f"mean background RMSE: {rmse_b.mean():.6f}")
    print(f"mean analysis RMSE  : {rmse_a.mean():.6f}")
