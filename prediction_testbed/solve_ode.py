
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

# Runge-Kuta integration
def runge_kuta4(rhs,state,dt,*args):
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2.,*args)
    k3 = rhs(state+k2*dt/2.,*args)
    k4 = rhs(state+k3*dt,*args)

    return state+(k1+2*k2+2*k3+k4)*dt/6.