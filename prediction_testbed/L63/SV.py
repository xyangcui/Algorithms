import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm, L96_tlm
from solve_ode import  runge_kuta4, rk4_nl_adm, rk4_nl_tlm
from fcst_utils import singular_vectors
import numpy as np
import pickle

dt   = 0.05  # time unit
K    = 40
F    = 8.
tmax = 3     # max integration
nt   = int(tmax/dt) 
bg_std = np.full(K,2.)  # perturb to generate initial state
nmember = 50   # number of members
obs_std = 1.5   # std of perturbed observation
# load ture value and observation
with open('database.pkl', 'rb') as f:
    data = pickle.load(f)
    obs  = data['obs']
    B    = data['B']
    R    = data['R']
# load background initial state
with open('background.pkl', 'rb') as f:
    initial_state_bk = pickle.load(f)
# load forecast ensemble [dim,ensemble,time]
with open('ensembleDA.pkl', 'rb') as f:
    initial_state = pickle.load(f)
_,nmember,ncase = initial_state.shape
# estimate analysis error vector.
AnVar = np.zeros((K,ncase),dtype=np.float64)
for icase in range(ncase):
    temp = initial_state[:,1:,icase]
    AnVar[:,icase] = np.std(temp,axis=1,ddof=1)
# projection matrix, currently set to identified matrix.
P = np.eye(K)
# a function to calculate total energy metrics.
def total_energy_norm(x):
    '''normalize x by its total energy.'''
    return x**2  
# function to integrate TLM. (only needs input as self-variable)
def TLM(x,zt,N,dt):
    for i in range(N):
        zt, x = rk4_nl_tlm(lambda x,y: L96_tlm(x,y),lambda x: L96(x,F),zt,x,dt)
    return x
# function to integrate ADM. (only needs input as self-variable)
def ADM(x,zt,N,dt):
    global K
    # step1: get trajectory
    zbase = np.zeros((K,N+1),dtype=np.float64)
    zbase[:,0] = zt
    for i in range(N):
        zbase[:,i+1] = runge_kuta4(lambda x: L96(x,F),zbase[:,i],dt)
    # step2: backward integrating
    for i in range(N, 0, -1):
        x = rk4_nl_adm(lambda x,y: L96_adm(x,y),lambda x: L96(x,F),zbase[:,i-1],x,dt)
    return x

def test_adjoint(zt, N, dt, F=8.0, num_tests=5, eps=1e-6, tol=1e-6):
    """
    测试 ADM 函数的正确性（伴随关系验证）
    
    参数:
        zt : 初始状态向量 (shape: K,)
        N  : 积分步数
        dt : 时间步长
        F  : L96 的强迫项（若你的 L96 需要）
        num_tests : 重复测试次数
        eps : 有限差分扰动幅度
        tol : 允许的误差容限
    """
    global K  # ADM 内部使用了全局 K，必须设置
    K = len(zt)
    
    # 定义非线性模式和切线性模式（有限差分）
    def nonlinear_forward(z0, steps):
        """前向积分 N 步，返回轨迹 (K, steps+1)"""
        traj = np.zeros((K, steps+1))
        traj[:, 0] = z0
        for i in range(steps):
            traj[:, i+1] = runge_kuta4(lambda x: L96(x, F), traj[:, i], dt)
        return traj
    
    for test_idx in range(num_tests):
        # 1. 随机生成扰动和伴随向量
        dx = np.random.randn(K)
        dy = np.random.randn(K)
        
        # 2. 计算非线性轨迹（参考）
        zbase = nonlinear_forward(zt, N)
        
        # 3. 计算 TLM(dx) 用有限差分
        z_plus = nonlinear_forward(zt + eps * dx, N)
        tlm_dx = (z_plus[:, -1] - zbase[:, -1]) / eps  # 终态扰动
        
        # 4. 计算伴随作用 ADM(dy)
        # 注意：ADM 的输入 x 是伴随变量（dy），zt 用于获取轨迹，N, dt 同前
        adm_dy = ADM(dy, zt, N, dt)   # 返回初始时刻的伴随变量
        
        # 5. 计算内积
        inner1 = np.dot(tlm_dx, dy)      # <TLM(dx), dy>
        inner2 = np.dot(dx, adm_dy)      # <dx, ADM(dy)>
        
        diff = np.abs(inner1 - inner2)
        print(f"Test {test_idx+1}: <TLM,dy> = {inner1:.10f}, <dx,ADM> = {inner2:.10f}, diff = {diff:.2e}")
        
        # 6. 断言检查
        assert diff < tol, f"伴随测试失败！差异 {diff} 超过容限 {tol}"
    
    print("所有测试通过！你的 ADM 实现正确。")



sv_t  = 1.  # 1 tu
sv_dt = 0.1 # 0.1 tu
icase = 0
sv = singular_vectors(m=K, 
                      nsv=K, 
                      scale=3, 
                      tol=1e-10, 
                      P=P, 
                      C0=lambda x: total_energy_norm(x),
                      CF=lambda x: total_energy_norm(x),
                      TLM = lambda x: TLM(x,initial_state[:,0,icase],int(sv_t/sv_dt),sv_dt),
                      ADM = lambda x: ADM(x,initial_state[:,0,icase],int(sv_t/sv_dt),sv_dt),
                      nmember=50,
                      Pa=AnVar[:,icase],
                      rescale=1.)