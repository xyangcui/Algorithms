import numpy as np
from math import sin
from solve_ode import runge_kuta4, rk4_nl_adm, rk4_nl_tlm

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

def L63_tlm(state,zt,*args):
    sigma = args[0] 
    rho   = args[1]
    beta  = args[2]

    dx,dy,dz = state
    x, y, z = zt
    f = np.zeros(3)
    f[0] = sigma*(dy-dx)
    f[1] = rho*dx-dy-x*dz-z*dx  #rho heating parameters
    f[2] = dx*y + x*dy- beta*dz

    return f

def L63_adm(lam,zt,*args):
    sigma = args[0] 
    rho   = args[1]
    beta  = args[2]

    lamx,lamy,lamz = lam
    x, y, z = zt

    f = np.zeros(3)
    # dz = y*dx + x*dy - beta*dz
    f[2] += -beta*lamz
    f[1] += x*lamz    
    f[0] += y*lamz
    # dy = (rho-z)*dx -dy -x*dz
    f[2] += -x*lamy
    f[1] += -lamy
    f[0] += (rho-z)*lamy
    # dx = simga*(dy - dx)
    f[1] += sigma*lamx
    f[0] += -sigma*lamx

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

def L96_tlm(state,z,*args):

    x = state
    n = len(state)      #dims
    f = np.zeros(n,dtype=np.float64)  

    if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e10):
        return np.zeros(n)
    # boundary
    f[0] = ((x[1] - x[n-2]) * z[n-1]+ (z[1] - z[n-2]) * x[n-1]- x[0])

    f[1] = ((x[2] - x[n-1]) * z[0]+ (z[2] - z[n-1]) * x[0]- x[1])

    f[n-1] = ((x[0] - x[n-3]) * z[n-2]+ (z[0] - z[n-3]) * x[n-2]- x[n-1])
    # inner
    for i in range(2, n-1):
        f[i] = ((x[i+1] - x[i-2]) * z[i-1]+ (z[i+1] - z[i-2]) * x[i-1]- x[i])

    return f

def L96_adm(lam_new,z):

    n = len(lam_new)      #dims
    lam_old = np.zeros(n,dtype=np.float64)  
    
    if np.any(np.isnan(lam_new)) or np.any(np.abs(lam_new) > 1e10):
        return np.zeros(n)
    # boundary
    # f[0] = (x[1] - x[n-2]) * z[n-1]+ (z[1] - z[n-2]) * x[n-1]- x[0]
    lam_old[1]   +=  z[n-1]*lam_new[0]
    lam_old[n-2] += -z[n-1]*lam_new[0]
    lam_old[n-1] += (z[1]-z[n-2])*lam_new[0]
    lam_old[0]   += -lam_new[0]
    #f[1] = (x[2] - x[n-1]) * z[0]+ (z[2] - z[n-1]) * x[0]- x[1]
    lam_old[2] += z[0]*lam_new[1]
    lam_old[n-1] += -z[0]*lam_new[1] 
    lam_old[0] += (z[2]-z[n-1])*lam_new[1]
    lam_old[1] += -lam_new[1]
    # f[n-1] = (x[0] - x[n-3]) * z[n-2]+ (z[0] - z[n-3]) * x[n-2]- x[n-1]
    lam_old[0] += z[n-2]*lam_new[n-1]
    lam_old[n-3] += -z[n-2]*lam_new[n-1] 
    lam_old[n-2] += (z[0]-z[n-3])*lam_new[n-1]
    lam_old[n-1] += -lam_new[n-1]    
    # inner
    for i in range(2, n-1):
        # f[i] = (x[i+1] - x[i-2]) * z[i-1]+ (z[i+1] - z[i-2]) * x[i-1]- x[i]
        lam_old[i+1] += z[i-1]*lam_new[i]
        lam_old[i-2] += -z[i-1]*lam_new[i] 
        lam_old[i-1] += (z[i+1]-z[i-2])*lam_new[i]
        lam_old[i] += -lam_new[i] 

    return lam_old

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


### functions to test ADM.
def forward_integrate(state, steps, dt, model_func, save_traj=True):
    """
    前向积分（非线性或切线性模式）
    
    参数:
        state : 初始状态向量 (shape: K,)
        steps : 积分步数
        dt    : 时间步长
        model_func : 模式算子，函数形式 f(state) 返回导数
        save_traj  : 是否保存完整轨迹 (默认 True)
    
    返回:
        final_state : 终态 (K,)
        traj        : 轨迹 (K, steps+1) 若 save_traj=True，否则为 None
    """
    K = len(state)
    if save_traj:
        traj = np.zeros((K, steps+1))
        traj[:, 0] = state
    else:
        traj = None
    
    current = state.copy()
    for i in range(steps):
        current = runge_kuta4(model_func, current, dt)
        if save_traj:
            traj[:, i+1] = current
    
    return current, traj

def adjoint_integrate(adjoint_state, traj, steps, dt, adjoint_func, model_func):
    """
    伴随积分（反向积分）
    
    参数:
        adjoint_state : 终态伴随变量 (shape: K,)
        traj          : 前向轨迹 (K, steps+1)，由 forward_integrate 返回
        steps         : 积分步数
        dt            : 时间步长
        adjoint_func  : 伴随算子，函数形式 f(adjoint, state) 返回伴随导数
        model_func    : 非线性模式算子（用于rk4_nl_adm内部，若不需要可设为 None，但这里保留）
    
    返回:
        init_adjoint : 初始时刻的伴随变量 (K,)
    """
    x = adjoint_state.copy()
    for i in range(steps, 0, -1):
        x = rk4_nl_adm(adjoint_func, model_func, traj[:, i-1], x, dt)
    return x

class ModelIntegrator:
    def __init__(self, model_func, adjoint_func, dt):
        """
        参数:
            model_func  : 非线性模式函数 f(state) 返回导数
            adjoint_func: 伴随算子函数 f(adjoint, state) 返回伴随导数
            dt          : 时间步长
        """
        self.model_func = model_func
        self.adjoint_func = adjoint_func
        self.dt = dt
    
    def forward(self, state, steps, save_traj=True):
        return forward_integrate(state, steps, self.dt, self.model_func, save_traj)
    
    def adjoint(self, adjoint_state, traj, steps):
        return adjoint_integrate(adjoint_state, traj, steps, self.dt, 
                                 self.adjoint_func, self.model_func)

def test_adjoint_integration(integrator, zt, steps, eps=1e-6, tol=1e-5):
    K = len(zt)
    # 前向轨迹（参考）
    _, traj = integrator.forward(zt, steps, save_traj=True)
    
    # 随机扰动和伴随向量
    dx = np.random.randn(K)
    dy = np.random.randn(K)
    
    # 有限差分 TLM
    _, traj_plus = integrator.forward(zt + eps*dx, steps, save_traj=True)
    tlm_dx = (traj_plus[:, -1] - traj[:, -1]) / eps
    
    # ADM
    adm_dy = integrator.adjoint(dy, traj, steps)
    
    inner1 = np.dot(tlm_dx, dy)
    inner2 = np.dot(dx, adm_dy)
    diff = np.abs(inner1 - inner2)
    print(f"<TLM,dy> = {inner1:.10f}, <dx,ADM> = {inner2:.10f}, diff = {diff:.2e}")
    assert diff < tol, f"伴随测试失败！diff={diff} > {tol}"
    print("测试通过")

if __name__ == "__main__":

    model = lambda x: L96(x, 8.)
    adjoint = lambda a, x: L96_adm(a, x)
    integrator = ModelIntegrator(model, adjoint, dt=0.05)
    zt = np.random.randn(40)
    test_adjoint_integration(integrator, zt, 3, eps=1e-6, tol=1e-5)