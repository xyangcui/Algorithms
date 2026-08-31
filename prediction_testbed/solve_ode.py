
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

  # previous two right hand sides wait for update.
lam_dt3, lam_dt2 = 0., 0.
def adams_bashforth_adm(lam_new, lam_dt1, dt, n):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, t, lam_dt3, lam_dt2
    if n == 0:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    elif n == 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0

    lam_old = lam_new + dt1*lam_dt1 + dt2*lam_dt2 + dt3*lam_dt3
    lam_dt3 = lam_dt2
    lam_dt2 = lam_dt1

    return lam_old

# Runge-Kuta integration
def runge_kuta4(rhs,state,dt,*args):
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2.,*args)
    k3 = rhs(state+k2*dt/2.,*args)
    k4 = rhs(state+k3*dt,*args)

    return state+(k1+2*k2+2*k3+k4)*dt/6.

def rk4_nl_tlm(tlm, nlm, zt, dzt, dt):

    k1 = nlm(zt)
    l1 = tlm(dzt, zt)

    z2  = zt  + 0.5 * dt * k1
    dz2 = dzt + 0.5 * dt * l1

    k2 = nlm(z2)
    l2 = tlm(dz2, z2)

    z3  = zt  + 0.5 * dt * k2
    dz3 = dzt + 0.5 * dt * l2

    k3 = nlm(z3)
    l3 = tlm(dz3, z3)

    z4  = zt  + dt * k3
    dz4 = dzt + dt * l3

    k4 = nlm(z4)
    l4 = tlm(dz4, z4)

    zt_new = zt + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

    dzt_new = dzt + dt / 6.0 * (l1 + 2*l2 + 2*l3 + l4)

    return zt_new, dzt_new


def rk4_nl_adm(adm,nlm,zt,lam_new, dt):
    '''
    RK4: adjoint model version.
    Input
      model: a class of baro model.
      zt: time n nonlinear state.
      lam_new: time n linear state.
      dt: time interval
    Output
      lam_old: adjoint variable.
    '''

    # 1. Reconstruct nonlinear RK4 trajectory
    k1 = nlm(zt)
    z2 = zt + 0.5 * dt * k1

    k2 = nlm(z2)
    z3 = zt + 0.5 * dt * k2

    k3 = nlm(z3)
    z4 = zt + dt * k3

    # procedure to procedure reverse.
    # dzt_new = dzt + dt/6. * (l1+ 2*l2 + 2*l3 + l4)
    lam_old = lam_new.copy()
    l1_ad  = dt/6.*lam_new
    l2_ad  = dt/6.*2 * lam_new
    l3_ad  = dt/6.*2 * lam_new
    l4_ad  = dt/6.*lam_new
    # l4 = model.bve_tlm(dz4, z4)
    dz4_ad = adm(l4_ad,z4)
    # dz4 = dzt + dt * l3
    lam_old += dz4_ad
    l3_ad  += dt*dz4_ad
    # l3 = model.bve_tlm(dz3, z3)
    dz3_ad  = adm(l3_ad,z3)
    # dz3 = dzt + 0.5dt* l2
    lam_old += dz3_ad
    l2_ad  += 0.5*dt*dz3_ad
    # l2 = model.bve_tlm(dz2,z2)
    dz2_ad  = adm(l2_ad,z2)
    # dz2 = dzt + 0.5dt* l1
    lam_old += dz2_ad
    l1_ad  += 0.5*dt*dz2_ad
    # l1 = model.bve_tlm(dzt, zt)
    lam_old += adm(l1_ad,zt)

    return lam_old 