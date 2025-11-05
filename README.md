# Damped SGM Analysis

This repository contains analysis scripts for data generated using the [Damped SGM simulator](https://github.com/amruthesht/damped-sgm-simulator). The plotting routines are specifically designed to generate figures for the manuscript: "Slow Relaxation and Landscape-Driven Dynamics in Viscous Ripening Foams" ([arXiv:2301.13400](https://arxiv.org/abs/2301.13400)).

## Analysis Scripts

- [`dr2ds_analysis_single_simulation.py`](analysis/pysrc/dr2ds_analysis_single_simulation.py): Script for 3N-dimensional configuration path analysis for a single ripening simulation.
- [`dr2ds_analysis_ensemble.py`](analysis/pysrc/dr2ds_analysis_ensemble.py): Script for computing ensemble averages over path analysis data (generated using [`dr2ds_analysis_single_simulation.py`](analysis/pysrc/dr2ds_analysis_single_simulation.py)) from multiple ripening simulations.
- [`dr2ds_a_i_analysis_single_simulation.py`](analysis/pysrc/dr2ds_a_i_analysis_single_simulation.py): Script for 3-dimensional particle-wise path analysis for a single ripening simulation.
- [`dr2ds_a_i_analysis_ensemble.py`](analysis/pysrc/dr2ds_a_i_analysis_ensemble.py): Script for computing radii-based ensemble averages over particle-wise path analysis data (generated using [`dr2ds_a_i_analysis_single_simulation.py`](analysis/pysrc/dr2ds_a_i_analysis_single_simulation.py)) from multiple simulations.
- [`dr2ds_RQ_analysis_single_simulation.py`](analysis/pysrc/dr2ds_RQ_analysis_single_simulation.py): Script for 3N-dimensional configuration path analysis for a single random quench (RQ) simulation.
- [`dr2ds_RQ_analysis_ensemble.py`](analysis/pysrc/dr2ds_RQ_analysis_ensemble.py): Script for computing ensemble averages over path analysis data (generated using [`dr2ds_RQ_analysis_single_simulation.py`](analysis/pysrc/dr2ds_RQ_analysis_single_simulation.py)) from multiple RQ simulations.

- [`QR_RQ_analysis.py`](analysis/pysrc/QR_RQ_analysis.py): Script for analyzing and comparing data from an ensemble of quasistatic ripening (QR) and random quench (RQ) simulations.

> Note: RQ simulations can be run using the Damped SGM simulator. However to execute multiple RQ simulations in an automated manner, code available [here](https://github.com/rar-ensemble/MIMSE) was used by modifying it slightly to match Damped SGM simulator's FIRE method ([FIRE.cpp](https://github.com/amruthesht/damped-sgm-simulator/blob/main/src/FIRE.cpp)).

- [`masonft.py`](analysis/pysrc/masonft.py): Module containing functions to compute the Mason number Fourier transform used for our rheological calculations.
- [`rheological_analysis.py`](analysis/pysrc/rheological_analysis.py): Script for computing the complex shear modulus $|G^*|$ from strain (positions) and stress time series data over multiple ripening simulations, using the formula described in the above manuscript.

## Plotting Scripts

- [`plot_figs.py`](plotting/pysrc/plot_figs.py): Script for generating all figures for the manuscript using simulation data from the simulator and post-analysis data generated using the above analysis scripts.
- [`plot_figs_App_SI.py`](plotting/pysrc/plot_figs_App_SI.py): Script for generating all figures for the Supplementary Information (SI) section of the manuscript.

> Note: Simulation data is not included in this repository and is available upon request.

Please contact [Amruthesh Thirumalaiswamy](mailto:amru@seas.upenn.edu) for any questions or clarifications regarding the usage of the above scripts or the simulator code.