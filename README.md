# DESMO
Decomposed sparse modal optimization (DESMO)


This repository contains the python/ipynb codes for the following paper:
Decomposed Sparse Modal Optimization: Interpretable Reduced-Order Modeling of Unsteady Flows


---

The repository contains four different version of DESMO with a few test cases.

**DESMO**: the main DESMO model. It contains three test cases:

- cylinder_flow: 2D flow over a cylinder at Re=100, leading to the periodic von Karman vortex street
- aneurysm: 3D blood flow in a cerebral aneurysm
- turbulent_channel: 2D slice of the turbulent channel flow from the Johns Hopkins Turbulence Databases


**DESMO_Fourier**: Uses Fourier-series for the temporal coefficients. It is only applicable for time-periodic flows. It contains two test cases:
- cylinder_flow
- aneurysm

**DESMO_AE**: Uses an autoencoder for finding the latent modes. It contains one test case, the flow over a cylinder.

**DESMO_SR**: Uses symbolic regression for fitting a dynamical systems model to the temporal coefficients in the form of dz/dt=f(z). It contains one test case, the flow over a cylinder. The dynamical systems fit is done in a postprocessing step, after fitting the DESMO model.
