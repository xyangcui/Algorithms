
toy_models: many toy models, include their nonlinear model, tangent linear model and adjoint model.

solve_ode: methods to solve pde.
- 3-order Adams-Bashforth.
- 4-order Runge Kuta.

DA_utilis: codes for procedures of data assimilation (variation, filter and nonlinear methods).
- Variation: 3DVar, **4DVar**
- Filter: EnKF, ETKF, **LETKF**, EAKF, EnSRF, Serial EnSRF
- Nonlinear: partical filter

fcst_utilis: codes to generate a forecast ensemble
- **singular vector**.
- breeding vector.
- NLLVs.

baro_utilis: a class for quasi-barotropic vorticity function. A intermediate model to practice.
- **bve_operator**: dynamics of the bve.
- **bve_tlm**: tlm of bve.
- **bve_adm**: adjoint propagator of bve.
- adams_bashforth: integrate using Adams-Bashforth 3.
- runge_kuta4: integrate using Runge Kuta 4.

l96_4dvar_pytorch: reconstructed 4DVar under the auto diffusion framework of PyTorch.