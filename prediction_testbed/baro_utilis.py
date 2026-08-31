
import numpy as np

from numpy import pi, cos, sin
from numpy.fft import fftshift, fftfreq, rfft2, irfft2, fft2, ifft2

### Function Definitions
def ft(phi):
    """Go from physical space to spectral space."""
    return fft2(phi, axes=(-2, -1))

def ift(psi):
    """Go from spectral space to physical space."""
    return ifft2(psi, axes=(-2,-1)).real

def ft_adjoint(psi):
    """
    Adjoint of: ft(phi) = fft2(phi) where phi is a real physical field.
    """
    N = psi.shape[-2] * psi.shape[-1]

    return N * ifft2(psi,axes=(-2, -1)).real

def ift_adjoint(phi):
    """
    Adjoint of: ift(psi) = ifft2(psi).real where phi is a real physical adjoint field.
    """
    N = phi.shape[-2] * phi.shape[-1]

    return fft2(phi,axes=(-2, -1)) / N

_prhs, _pprhs  = 0.0, 0.0  # previous two right hand sides
def adams_bashforth(zt, rhs, dt):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, t, _prhs, _pprhs
    if step == 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif step == 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newzt = zt + dt1*rhs + dt2*_prhs + dt3*_pprhs
    _pprhs = _prhs
    _prhs  = rhs
    return newzt

prhs_z, pprhs_z, prhs_dz, pprhs_dz  = 0., 0., 0., 0.  # previous two right hand sides
def adams_bashforth_tlm(dzt, zt, rhsz, rhsdz, dt):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, t, prhs_z, pprhs_z, prhs_dz, pprhs_dz
    if step == 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif step == 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newzt = zt + dt1*rhsz + dt2*prhs_z + dt3*pprhs_z
    newdzt= dzt+ dt1*rhsdz + dt2*prhs_dz + dt3*pprhs_dz

    pprhs_z = prhs_z; prhs_z = rhsz
    pprhs_dz = prhs_dz; prhs_dz = rhsdz

    return newzt, newdzt

zt_dt2, zt_dt3 = 0. ,0.
def adams_bashforth_adm(adm,lam_new, lam, dt, n):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global lam_dt3, lam_dt2, zt_dt2, zt_dt3
    if n == 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif n == 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    lam[n]  = lam_dt1.copy()
    lam_new += dt1*adm(lam_dt1,zt_dt1)
    if n >1:
    lam_dt2 += dt2*adm(lam_dt2,zt_dt2)
    zt_dt2 = zt_dt3

    lam_dt3 += dt3*adm(lam_dt3,zt_dt3)
    zt_dt3 = zt_dt1

    return lam_new

# Runge-Kuta integration
def runge_kuta4(rhs,state,dt,*args):
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2.,*args)
    k3 = rhs(state+k2*dt/2.,*args)
    k4 = rhs(state+k3*dt,*args)

    return state+(k1+2*k2+2*k3+k4)*dt/6.

def rk4_nl_tlm(model, zt, dzt, dt, forcet):

    k1 = model.bve_operator(zt, forcet)
    l1 = model.bve_tlm(dzt, zt)

    z2  = zt  + 0.5 * dt * k1
    dz2 = dzt + 0.5 * dt * l1

    k2 = model.bve_operator(z2, forcet)
    l2 = model.bve_tlm(dz2, z2)

    z3  = zt  + 0.5 * dt * k2
    dz3 = dzt + 0.5 * dt * l2

    k3 = model.bve_operator(z3, forcet)
    l3 = model.bve_tlm(dz3, z3)

    z4  = zt  + dt * k3
    dz4 = dzt + dt * l3

    k4 = model.bve_operator(z4, forcet)
    l4 = model.bve_tlm(dz4, z4)

    zt_new = zt + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

    dzt_new = dzt + dt / 6.0 * (l1 + 2*l2 + 2*l3 + l4)

    return zt_new, dzt_new

def rk4_nl_adm(model,zt,lam_new, dt,forcet):
    '''
    RK4: adjoint model version.
    Input
      model: a class of baro model.
      zt: time n nonlinear state.
      lam_new: time n linear state.
      dt: time interval
      forcet: forcing.
    Output
      lam_old: adjoint variable.
    '''

    # 1. Reconstruct nonlinear RK4 trajectory
    k1 = model.bve_operator(zt, forcet)
    z2 = zt + 0.5 * dt * k1

    k2 = model.bve_operator(z2, forcet)
    z3 = zt + 0.5 * dt * k2

    k3 = model.bve_operator(z3, forcet)
    z4 = zt + dt * k3

    # procedure to procedure reverse.
    # dzt_new = dzt + dt/6. * (l1+ 2*l2 + 2*l3 + l4)
    lam_old = lam_new.copy()
    l1_ad  = dt/6.*lam_new
    l2_ad  = dt/6.*2 * lam_new
    l3_ad  = dt/6.*2 * lam_new
    l4_ad  = dt/6.*lam_new
    # l4 = model.bve_tlm(dz4, z4)
    dz4_ad = model.bve_adm(l4_ad,z4)
    # dz4 = dzt + dt * l3
    lam_old += dz4_ad
    l3_ad  += dt*dz4_ad
    # l3 = model.bve_tlm(dz3, z3)
    dz3_ad  = model.bve_adm(l3_ad,z3)
    # dz3 = dzt + 0.5dt* l2
    lam_old += dz3_ad
    l2_ad  += 0.5*dt*dz3_ad
    # l2 = model.bve_tlm(dz2,z2)
    dz2_ad  = model.bve_adm(l2_ad,z2)
    # dz2 = dzt + 0.5dt* l1
    lam_old += dz2_ad
    l1_ad  += 0.5*dt*dz2_ad
    # l1 = model.bve_tlm(dzt, zt)
    lam_old += model.bve_adm(l1_ad,zt)

    return lam_old  

    

class BARO_VORT:
    """beta plane barotropic vorticity model.
   By James Penn, Geoffrey K. Vallis

   This is meant as a self-contained, simple to use barotropic code. 
   A more complete python code is to be found at pyqg https://github.com/pyqg/pyqg]
   or a fast fortran code on the web site of K. Shafer Smith (Courant).
   
This script uses a pseudospectral method to solve the barotropic vorticity
equation in two dimensions

    D/Dt[ω] = forcing - dissipation                                                            (1)

where ω = ξ + f.  ξ is local vorticity ∇ × u and f is global rotation.

Assuming an incompressible two-dimensional flow u = (u, v),
the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)

    u = ∂/∂y[ψ]         v = -∂/∂x[ψ]                                        (2)

and therefore local vorticity can be given as a Poisson equation

    ξ = ∆ψ                                                                  (3)

where ∆ is the laplacian operator.  Since ∂/∂t[f] = 0 equation (1) can be
written in terms of the local vorticity

        D/Dt[ξ] + u·∇f = 0
    =>  D/Dt[ξ] = -vβ                                                       (4)

using the beta-plane approximation f = f0 + βy.  This can be written entirely
in terms of the streamfunction and this is the form that will be solved
numerically.

    D/Dt[∆ψ] = -β ∂/∂x[ψ]                                                   (5)

The spectral method defines ψ as a Fourier sum

    ψ = Σ A(t) exp(i (kx + ly))

and as such spatial derivatives can be calculated analytically

    ∂/∂x[ψ] = ikψ       ∂/∂y[ψ] = ilψ

The pseudospectral method will use the analytic derivatives to calculate
values for (u, v) which will then be used to evaluate nonlinear terms.

This version has no forcing and a high wavenumber Smith filter
which replaces hyperviscosity. 

It can be de-aliased. 

"""
    def __init__(self, nx=256, ny=256, 
                       Lx=1.0, Ly=1.0, 
                       ubar=0.00, 
                       beta=12.0,
                       n_diss=2.0, 
                       tau=0.1,
                       dt=0.01,
                       r_rayleigh=(1./50000.)/np.sqrt(10.),
                       forcing_amp_factor=100.0/np.sqrt(1.),
                       filter_exp=8.0, kcut=30.0,
                       speedup_at_c=0.4, slowdn_at_c=0.6,allow_speedup=True):
        # input parameter
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.ubar = ubar
        self.beta = beta
        self.n_diss = n_diss
        self.tau = tau
        self.r_rayleigh = r_rayleigh
        self.forcing_amp_factor = forcing_amp_factor
        self.allow_speedup = allow_speedup
        self.speedup_at_c = speedup_at_c
        self.slowdn_at_c = slowdn_at_c
        self.filter_exp = filter_exp
        self.kcut = kcut
        self.dt   = dt
        self.amp  = 0.
        # build-in parameter
        self._setup_derived_parameters()

    def _setup_derived_parameters(self):
        # physical space
        self.nl = self.ny
        self.nk = self.nx

        self.y = np.linspace(0, self.Ly, num=self.ny)
        self.y_arr = np.flipud(np.tile(self.y, (self.nx, 1)).transpose())
        self.xx = np.linspace(0, self.Lx, num=self.nx)
        self.yy = 1. - np.linspace(0, self.Ly, num=self.ny)
        # spatial intervals
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        # spectral interval
        self.dk = 2.0 * pi / self.Lx
        self.dl = 2.0 * pi / self.Ly
        # wavenumber
        #self.k = self.dk * np.arange(0, self.nk, dtype=np.float64)[np.newaxis, :]
        self.k = self.dk*fftfreq(self.nk, d=1.0/self.nl)[None, :]
        self.l = self.dl*fftfreq(self.nl, d=1.0/self.nl)[:, None]
        # total wavenumber k^2+l^2
        self.ksq = self.k**2 + self.l**2
        self.ksq[self.ksq == 0] = 1.0          # avoid dividing 0
        self.rksq = 1.0 / self.ksq             # 1/(k²+l²)
        self.ik = 1j * self.k
        self.il = 1j * self.l
        # dissipation parameters
        # reference scale
        self.nu = ((self.Lx / (np.floor(self.nx / 3) * 2.0 * pi)) ** (2 * self.n_diss)) / self.tau
        # spectral filter
        # wavenumber
        self.wvx = np.sqrt((self.k * self.dx)**2 + (self.l * self.dy)**2)
        self.spectral_filter = np.exp(-23.6 * (self.wvx - 0.65 * pi)**4)
        # keep（wvx <= 0.65π）as 1
        self.spectral_filter[self.wvx <= 0.65 * pi] = 1.0

    ## build-in function
    def grad(self,phit):
        """Returns the spatial derivatives of a Fourier transformed variable.
    Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
        phixt = self.ik*phit        # d/dx F[φ] = ik F[φ]
        phiyt = self.il*phit        # d/dy F[φ] = il F[φ]
        return (phixt, phiyt)

    def grad_adjoint(self,phixt,phiyt):
        """Returns the spatial derivatives of a Fourier transformed variable.
    Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
        return (np.conj(self.ik))*phixt + (np.conj(self.il))*phiyt

    def velocity(self,psit):
        """Returns the velocity field (u, v) from F[ψ]."""
        psixt, psiyt = self.grad(psit)
        psix = ift(psixt)    # v =   ∂/∂x[ψ]
        psiy = ift(psiyt)    # u = - ∂/∂y[ψ]
        return (-psiy, psix)

    def anti_alias(self,phit):
        """Set the coefficients of wavenumbers > k_mask to be zero."""
        k_mask = (8./9.)*(self.nk+1)**2.
        phit[(np.abs(self.ksq/(self.dk*self.dk)) >= k_mask)] = 0.0

    def spectral_variance(self,phit):
        var_density = 2.0 * np.abs(phit)**2 / (self.nx*self.ny)
        var_density[:,0] /= 2
        var_density[:,-1] /= 2
        return var_density.sum()

    def high_wn_filter(self,phit):
        """Applies the high wavenumber filter of smith et al 2002"""
        filter_dec = -np.log(1.+2.*pi/self.nk)/((self.nk-self.kcut)**self.filter_exp)
        filter_idx = np.abs(self.ksq/(self.dk*self.dk)) >= self.kcut**2.
        phit[filter_idx] *= np.exp(filter_dec*(np.sqrt(self.ksq[filter_idx]/(self.dk*self.dk))-self.kcut)**self.filter_exp)

    def courant_number(self,psix, psiy, dt):
        """Calculate the Courant Number given the velocity field and step size."""
        maxu = np.max(np.abs(psiy))
        maxv = np.max(np.abs(psix))
        maxvel = maxu + maxv
        return maxvel*dt/self.dx

    def initial_state(self):
        z = np.zeros((self.ny, self.nx), dtype=np.float64)
        zt = np.zeros((int(self.nl), int(self.nk)), dtype=np.complex128)

        ### Initial Condition
        # The McWilliams Initial Condition from [McWilliams - J. Fluid Mech. (1984)]
        #ck   = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
        ck   = 0.001/(self.ksq + (self.ksq - 3200.)**2)
        piit = np.random.randn(*self.ksq.shape)*ck + 1j*np.random.randn(*self.ksq.shape)*ck

        pii  = ift(piit)
        pii  = pii - pii.mean()
        piit = ft(pii)
        KE   = self.spectral_variance(piit*np.sqrt(self.ksq)*self.spectral_filter)

        qit = -self.ksq * piit / np.sqrt(KE)
        qi = ift(qit)
        z = qi
        # initialise the transformed ζ
        zt = ft(z)
        self.anti_alias(zt)
        z = ift(zt)
        # calc a reasonable forcing amplitude
        self.amp = forcing_amp_factor* np.max(np.abs(qi)) 
        return z, zt
    
    def processing(self,zt):
        # 0.1 calculate derivatives in spectral space (spectral)
        psit = -self.rksq * zt           # F[ψ] = - F[ζ] / (k^2 + l^2)
        psixt, psiyt = model.grad(psit)
        # 0.2 transform back to physical space for courant number
        psix = ift(psixt)
        psiy = ift(psiyt)
        # 0.3 calculate the size of timestep that can be taken
        # (assumes a domain where dx and dy are of the same order)
        c = model.courant_number(psix, psiy, self.dt)
        if c >= self.slowdn_at_c:
            print('DEBUG: Courant No > 0.8, reducing timestep')
            self.dt = 0.9*self.dt
        elif c < self.speedup_at_c and self.allow_speedup:
            self.dt = 1.1*self.dt  

        return c, self.dt   

    def forcing_operator(self,shapez):
        '''
    randomized forcing in physical world and transfer to spectral domain.
    Input
      ksq: square total wavenumber. k^2+l^2
       dk: spatial interval in spectral domain
      amp: the amplitude of forcing.
    Output
      forcet: randomized forcing. in spectral domain.
    '''
        # set forcing at physical space
        force = self.amp * (np.random.random(shapez) - 0.5)
        # transfer to spectral space
        forcet = ft(force)
        # get magnitude of wavenumber
        kmag = np.sqrt(self.ksq) / self.dk
        # only absorb forcing to 14~20.
        idx = ((kmag > 14.0)& (kmag < 20.0))
        forcet = forcet*idx

        return forcet  

    def hyperviscosity(self,zt,dt):
        '''
    A backward Euler, use hyperviscosity to urge high energy and wavenumber waves.

    Input
      zt:  vorticity in spectral space.
      ksq: square of total wavenumber. like: k^2+l^2
      nu:  hyperviscosity parameter
      dt:  time interval
      n_diss: order of dissipation. if n_diss=2, it represents a 4-order hyperviscosity. ∇^4
    Output
      -nu∇^4(zt) spectral space 
    '''
        deln = 1.0 / (1.0 + self.nu*self.ksq**self.n_diss*dt)
        return deln * zt 

    def bve_operator(self,zt,forcing=None):
        """
    Barotropic vorticity equation tendency operator.
    Equation like
    d(zeta_hat)/dt = -FFT[J(psi,zeta) + ubar*zeta_x] -beta*psi_x_hat -r*zeta_hat +forcing_hat
    where nabla^2 psi = zeta.

    Input
      zt: vorticity in spectral domain.
      rksq: 
      beta: beta parameter. f=f0+beta*y
      ubar: background zonal wind.
      r_rayleigh: a parameter to control rayleigh damping
      forcing: forcing. default is no forcing.
    
    Output
      rhs: the distance of evolution
    """
        # use vorticity obtain phi.
        psit = -self.rksq * zt
        # gradients of phi and vorticity
        psixt, psiyt = self.grad(psit)
        zxt, zyt = self.grad(zt)
        # transfer to physical space both phi and vorcity
        psix = ift(psixt)
        psiy = ift(psiyt)
        zx = ift(zxt)
        zy = ift(zyt)
        # calculate the Jacob term in physical space and transfer to spectral space
        jac = (psix * zy- psiy * zx+ self.ubar * zx)
        jact = ft(jac)
        # add forcing in spectral space.
        if (forcing == None).any():
            rhs = -jact -self.beta*psixt -self.r_rayleigh*zt 
        else:
            rhs = -jact -self.beta*psixt -self.r_rayleigh*zt + forcet

        return rhs  

    def bve_tlm(self,dzt,zt):
        """
    Tangent linear model of Barotropic vorticity equation tendency operator.
    Input
      zt: vorticity in spectral domain.
     dzt: vorticity perturbation in spectral domain. 
      beta: beta parameter. f=f0+beta*y
      ubar: background zonal wind.
      r_rayleigh: a parameter to control rayleigh damping
    
    Output
      rhs: the distance of evolution
    """
        # vorticity to obtain phi
        psidt = -self.rksq * dzt  # del_phi in spectral domain
        psit  = -self.rksq * zt   # phi in spectral domain
        # gradients of phi and vorticity in spectral space
        psixdt, psiydt = self.grad(psidt) # del_phi
        zxdt, zydt     = self.grad(dzt)   # del_vor
        psixt, psiyt   = self.grad(psit)  # phi
        zxt, zyt       = self.grad(zt)    # vor
        # transfer to physical space both phi and vorcity
        psixd = ift(psixdt); psiyd = ift(psiydt) # del_phi
        zxd = ift(zxdt); zyd = ift(zydt)         # del_vor
        psix = ift(psixt); psiy = ift(psiyt)     # phi
        zx = ift(zxt); zy = ift(zyt)             # vor
        # the jacob term in physical space and transfer to spectral space.
        jacd = (psix * zyd- psiy * zxd+ self.ubar * zxd)
        jac  = (psixd * zy- psiyd * zx)
        jacdt= ft(jacd)
        jact = ft(jac)

        rhs = -jacdt -jact -self.beta*psixdt -self.r_rayleigh*dzt 

        return rhs

    def bve_tlm_propagator(self,models,zt,dzt,forcet):
        '''
        One step of integrating TLM with RK4.
        Input
          models: a class of model.
          zt: nonlinear state.
          dzt: linear state
        Output
          zt: nonliear state
          dzt: linear state.
        '''

        # RK4 NLM + TLM
        zt, dzt = rk4_nl_tlm(models,zt,dzt,self.dt,forcet)
        # same post-processing for BOTH
        zt = self.hyperviscosity(zt, dt)
        dzt = self.hyperviscosity(dzt, dt)

        self.anti_alias(zt)
        self.anti_alias(dzt)

        return zt, dzt

    def bve_adm(self,rhs_ad,zt):
        """
    Adjoint model of Barotropic vorticity equation tendency operator.
    Input
        rhs_ad: dynamical term passed from time integration.
        zt: time n nonlinear state.
    Output
        lam_old: adjoint variable
    """
        psit  = -self.rksq * zt   # phi in spectral domain
        # grad phi in spectral and physical space.
        psixt, psiyt   = self.grad(psit) 
        psix = ift(psixt); psiy = ift(psiyt)    
        # grad z in spectral and physical space.
        zxt, zyt   = self.grad(zt) 
        zx = ift(zxt); zy = ift(zyt)

        # define all appeared adjoint variable.
        jacdt_ad  = np.zeros_like(rhs_ad); jact_ad = np.zeros_like(rhs_ad) # spectra
        psixdt_ad = np.zeros_like(rhs_ad); lam_old = np.zeros_like(rhs_ad) # spectra
        psiydt_ad = np.zeros_like(rhs_ad)   # spectra
        jac_ad = np.zeros_like(zx); jacd_ad = np.zeros_like(zx)   # physical
        psixd_ad = np.zeros_like(zx); psiyd_ad = np.zeros_like(zx) # physical
        zyd_ad = np.zeros_like(zx); zxd_ad = np.zeros_like(zx)     # physical
        zxdt_ad = np.zeros_like(rhs_ad); zydt_ad = np.zeros_like(rhs_ad) # spectra
        psidt_ad = np.zeros_like(psixdt_ad)   # spectra

        # rhs = -jacdt -jact -self.beta*psixdt -self.r_rayleigh*dzt 
        jacdt_ad += -rhs_ad
        jact_ad  += -rhs_ad
        psixdt_ad+= -self.beta*rhs_ad
        lam_old  += -self.r_rayleigh*rhs_ad
        # jact = ft(jac)
        jac_ad += ft_adjoint(jact_ad)
        # jacdt= ft(jacd)
        jacd_ad += ft_adjoint(jacdt_ad)
        # jac  = (psixd * zy- psiyd * zx)
        psixd_ad += zy * jac_ad
        psiyd_ad += -zx * jac_ad
        # jacd = (psix * zyd- psiy * zxd+ self.ubar * zxd)
        zyd_ad += psix*jacd_ad
        zxd_ad += -psiy*jacd_ad + self.ubar*jacd_ad
        # zxd = ift(zxdt); zyd = ift(zydt)   
        zxdt_ad += ift_adjoint(zxd_ad)
        zydt_ad += ift_adjoint(zyd_ad)
        # psixd = ift(psixdt); psiyd = ift(psiydt)
        psixdt_ad += ift_adjoint(psixd_ad)
        psiydt_ad += ift_adjoint(psiyd_ad)
        # zxdt, zydt = self.grad(dzt)
        lam_old  += self.grad_adjoint(zxdt_ad, zydt_ad)  
        # psixdt, psiydt = self.grad(psidt)
        psidt_ad += self.grad_adjoint(psixdt_ad,psiydt_ad)     
        # psidt = -self.rksq * dzt
        lam_old += -self.rksq * psidt_ad

        return lam_old

    def bve_adm_propagator(self,models,nstep,z_base,lam,forcet):
        '''
        One step integration of ADM.
        Input
          models: a class of model.
          nstep: current steps of integration.
          z_base[nstep,ndims]: till now nonlinear states.
          lam: current linear state.
          forcet: forcing
        Output
          lam: adjoint variable
        '''
        for n in range(nstep - 1, -1, -1):
            zt = ft(z_base[n])

            models.anti_alias(lam)
            lam = models.hyperviscosity(lam,self.dt)
            lam = rk4_nl_adm(models,zt,lam,self.dt,forcet)

        return lam

if __name__ == "__main__":

    nx = 256
    ny = 256
    Lx = 1.0
    Ly = 1.0
    ubar = 0.00
    beta = 12.0
    n_diss = 2.0
    tau = 0.1
    r_rayleigh = (1. / 50000.) / np.sqrt(10.)
    forcing_amp_factor = 100.0 / np.sqrt(1.)
    filter_exp = 8.0
    kcut = 30.0
    SPEEDUP_AT_C = 0.4
    SLOWDN_AT_C  = 0.6
    ALLOW_SPEEDUP = False
    dt = 0.02#0.4 * 16.0 / nx
    model = BARO_VORT(
        nx=nx, 
        ny=ny, 
        Lx=Lx, 
        Ly=Ly,
        ubar=ubar, 
        beta=beta, 
        n_diss=n_diss, 
        tau=tau,
        dt=dt,
        r_rayleigh=r_rayleigh, 
        forcing_amp_factor=forcing_amp_factor,
        filter_exp=filter_exp, 
        kcut=kcut,
        speedup_at_c=SPEEDUP_AT_C, 
        slowdn_at_c=SLOWDN_AT_C,
        allow_speedup=ALLOW_SPEEDUP
    )

    ## SETUP
    tmax = 1; t = 0.0; step = 0

    # initial state
    z, zt = model.initial_state()
    # array to store value
    vor = np.empty([int(tmax/dt)+1,z.shape[0],z.shape[1]])

    # 2. apply fixed forcing in spectral space by exciting certain wavenumbers
    forcet = model.forcing_operator(z.shape)

    ## RUN THE SIMULATION
    while t <= tmax:
        # transfer to physical space
        vor[step] = ift(zt)
        # 1. processing for adjust dt
        c, _ = model.processing(zt)
        # 3. dynamics and integration
        # adms bashforth3
        rhs = model.bve_operator(zt,forcet)
        zt  = adams_bashforth(zt, rhs, dt)
        # runge kuta4
        #rhs = lambda z: model.bve_operator(z, forcet)
        #zt  = runge_kuta4(rhs, zt, dt)
        # 4. use hyperviscosity to reduce high wavenumber
        zt = model.hyperviscosity(zt,dt)
        # 5. anti_alias
        model.anti_alias(zt)
        # diagnosis
        print('[{:5d}] {:.2f} Max z: {:2.2f} c={:.2f} dt={:.3f}'.format(step, t, np.max(vor[step]), c, dt))

        t = t + dt
        step = step + 1

    # test TLM
    def test_tlm(model, z0, dt, tmax, forcet,alpha=0.1, ntest=15, seed=1234):

        rng = np.random.default_rng(seed)
        nstep = int(round(tmax / dt))
        # Initial perturbation
        zt0 = ft(z0)
        dz0 = rng.standard_normal(z0.shape)

        dz0 -= dz0.mean()
        dz0 /= np.linalg.norm(dz0)
        dzt0 = ft(dz0)

        zt = zt0.copy()
        dzt = dzt0.copy()

        z_base = np.empty((nstep + 1,) + z0.shape)
        dz_tlm = np.empty_like(z_base)

        z_base[0] = z0
        dz_tlm[0] = dz0

        # 1. Integrate TLM
        for n in range(nstep):
            # RK4 NLM + TLM
            zt, dzt = rk4_nl_tlm(model,zt,dzt,dt,forcet)
            # same post-processing for BOTH
            zt = model.hyperviscosity(zt, dt)
            dzt = model.hyperviscosity(dzt, dt)

            model.anti_alias(zt)
            model.anti_alias(dzt)

            z_base[n + 1] = ift(zt)
            dz_tlm[n + 1] = ift(dzt)
        
        tlm_final = dz_tlm[-1]
        tlm_norm = np.linalg.norm(tlm_final)
        # 2. Perturbed nonlinear integrations
        R = np.zeros(ntest)

        for k in range(ntest):
            eps = alpha**k
            zt_p = ft(z0 + eps * dz0)

            for n in range(nstep):
                rhs = lambda z: model.bve_operator(z, forcet)
                zt_p = runge_kuta4(rhs,zt_p,dt)
                zt_p = model.hyperviscosity(zt_p,dt)
                model.anti_alias(zt_p)

            # nonlinear perturbation at final time
            dz_nl = ift(zt_p) - z_base[-1]
            nonlinear_norm = np.linalg.norm(dz_nl)
            linear_norm = eps * tlm_norm

            R[k] = nonlinear_norm / linear_norm

        print("TLM ratio test")
        for k, r in enumerate(R):
            print(f"alpha^{k:2d} = {alpha**k:.3e}    "f"R = {r:.15e}")

        return R, dz0, dz_tlm

    ### test ADM
    def test_bve_adjoint(model, zt):
        '''
            Test adjoint propagator itself.
        '''
        rng = np.random.default_rng(1234)
        # Generate physically valid spectral fields
        dzt_phys = rng.standard_normal(zt.shape)
        lam_phys = rng.standard_normal(zt.shape)
        # optional: zero mean if k=0 mode is excluded
        dzt_phys -= dzt_phys.mean()
        lam_phys -= lam_phys.mean()
        dzt_phys /= np.linalg.norm(dzt_phys)
        lam_phys /= np.linalg.norm(lam_phys)
        dzt = ft(dzt_phys)
        lam = ft(lam_phys)
        # TLM
        Ldzt = model.bve_tlm(dzt, zt)
        # ADM
        LTlam = model.bve_adm(lam, zt)
        # Dot-product test
        lhs = np.real(np.vdot(Ldzt, lam))
        rhs = np.real(np.vdot(dzt, LTlam))

        relerr = abs(lhs - rhs) / max(abs(lhs),abs(rhs),1e-30)

        print("adjoint propagator test")
        print("lhs =", lhs)
        print("rhs =", rhs)
        print("difference =", lhs-rhs)
        print("relative error =", relerr)

    def test_adjoint_rk4(model, z0, dt, tmax, forcet,alpha=0.1, ntest=15, seed=1234):
        '''Test <Mu, v> == <M*v, u>'''
        rng = np.random.default_rng(seed)

        nstep = int(round(tmax / dt))
        # Initial perturbation
        zt0 = ft(z0)
        dz0 = rng.standard_normal(z0.shape)

        dz0 -= dz0.mean()
        dz0 /= np.linalg.norm(dz0)
        dzt0 = ft(dz0)

        zt = zt0.copy()
        dzt = dzt0.copy()

        z_base = np.empty((nstep + 1,) + z0.shape)
        dz_tlm = np.empty_like(z_base)
        z_adm  = np.empty_like(z_base)

        z_base[0] = z0
        dz_tlm[0] = dz0

        # 1. Integrate TLM and real NLM
        for n in range(nstep):
            # RK4 NLM + TLM
            zt, dzt = rk4_nl_tlm(model,zt,dzt,dt,forcet)
            # same post-processing for BOTH
            zt = model.hyperviscosity(zt, dt)
            dzt = model.hyperviscosity(dzt, dt)

            model.anti_alias(zt)
            model.anti_alias(dzt)

            z_base[n + 1] = ift(zt)
            dz_tlm[n + 1] = ift(dzt)  
        # 2. Integrate ADM
        # TLM final perturbation
        Mu = dzt.copy()
        # terminal adjoint
        # random terminal adjoint variable
        rng = np.random.default_rng(5678)
        lam_phys = rng.standard_normal(z0.shape)
        lam_phys -= lam_phys.mean()
        lam_phys /= np.linalg.norm(lam_phys)
        lam = ft(lam_phys)
        v = lam
        #lam = Mu.copy()
        z_adm[-1] = ift(lam)
        for n in range(nstep - 1, -1, -1):
            zt = ft(z_base[n])

            model.anti_alias(lam)
            lam = model.hyperviscosity(lam,dt)
            lam = rk4_nl_adm(model,zt,lam,dt,forcet)

            z_adm[n] = ift(lam)

        # 4. parameter
        u = ft(dz0)
        Mstar_Mu = ft(z_adm[0])
        lhs = np.real(np.vdot(Mu, v))
        rhs = np.real(np.vdot(u, Mstar_Mu))

        err = abs(lhs-rhs) / max(abs(lhs), abs(rhs), 1e-30)   

        print("adjoint rk4 test")
        print("lhs =", lhs)
        print("rhs =", rhs)
        print("relative error =", err)   

    def test_adjoint_ab3(model, z0, dt, tmax, forcet,alpha=0.1, ntest=15, seed=1234):
        '''Test <Mu, v> == <M*v, u>'''
        rng = np.random.default_rng(seed)

        nstep = int(round(tmax / dt))
        # Initial perturbation
        zt0 = ft(z0)
        dz0 = rng.standard_normal(z0.shape)

        dz0 -= dz0.mean()
        dz0 /= np.linalg.norm(dz0)
        dzt0 = ft(dz0)

        zt = zt0.copy()
        dzt = dzt0.copy()

        z_base = np.empty((nstep + 1,) + z0.shape)
        dz_tlm = np.empty_like(z_base)
        z_adm  = np.empty_like(z_base)

        z_base[0] = z0
        dz_tlm[0] = dz0

        # 1. Integrate TLM and real NLM
        for n in range(nstep):
            # RK4 NLM + TLM
            rhsz = model.bve_operator(zt,forcet)
            rhsdz= model.bve_tlm(dzt,zt)
            zt, dzt = adams_bashforth_tlm(dzt,zt,rhsz,rhsdz,dt)
            # same post-processing for BOTH
            zt = model.hyperviscosity(zt, dt)
            dzt = model.hyperviscosity(dzt, dt)

            model.anti_alias(zt)
            model.anti_alias(dzt)

            z_base[n + 1] = ift(zt)
            dz_tlm[n + 1] = ift(dzt)  
        # 2. Integrate ADM
        # TLM final perturbation
        Mu = dzt.copy()
        # terminal adjoint
        # random terminal adjoint variable
        rng = np.random.default_rng(5678)
        lam_phys = rng.standard_normal(z0.shape)
        lam_phys -= lam_phys.mean()
        lam_phys /= np.linalg.norm(lam_phys)
        lam = ft(lam_phys)
        v = lam
        #lam = Mu.copy()
        z_adm[-1] = ift(lam)
        for n in range(nstep - 1, -1, -1):
            zt = ft(z_base[n])

            model.anti_alias(lam)
            lam = model.hyperviscosity(lam,dt)
            lam_rhs = model.bve_adm(lam,zt)
            lam = adams_bashforth_adm(lam,lam_rhs,dt,n)

            z_adm[n] = ift(lam)

        # 4. parameter
        u = ft(dz0)
        Mstar_Mu = ft(z_adm[0])
        lhs = np.real(np.vdot(Mu, v))
        rhs = np.real(np.vdot(u, Mstar_Mu))

        err = abs(lhs-rhs) / max(abs(lhs), abs(rhs), 1e-30)   

        print("adjoint ab3 test")
        print("lhs =", lhs)
        print("rhs =", rhs)
        print("relative error =", err)        

    #test_tlm(model,vor[0],dt,tmax,forcet)
    #test_bve_adjoint(model, zt)
    #test_adjoint_rk4(model,vor[0],dt,tmax,forcet)
    test_adjoint_ab3(model,vor[0],dt,tmax,forcet)