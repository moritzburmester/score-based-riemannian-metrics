# Riemannian Metrics from Score-based  Models 
GitHub repository for my Master Thesis on Riemannian Metrics from Score-based  Models


## Overview 
This repository contains the code necessary to reproduce the experiments from my thesis. Specifically, I investigated the behavior of different Riemannian metrics from literature, as well as proposed a new metric that interpolates between manifold-aware and density-aware terms. The manifold-aware term is composed of the Jacobian of the score function, where as the density-aware term is the magnitude of the score function. The Jacobian term guides geodesics to move tangentially to the underlying data manifold, where as the magnitude term ideally pulls the geodesic towards higher density regions.

## Checkpoints and Geodesics
Checkpoints for each model and geodesics saved as .pt files can be found in the respective experiment folders. 

## Repository Structure 
```
├── assets 
├── _model_dependencies/   # scripts and modules to train, evaluate, and sample from score-based  models
├── experiments_toy_datasets/       # geodesics under different metrics for the toy datasets (circle, s-curve, swiss-roll/spiral, ucg, wcg, two moons)
├── experiments_urc/                # geodesics on the uniform rotated characters (URC) dataset
├── experiments_mbm/         # geodesics in the stable  latent space
├── tutorial.ipynb                  # notebook to optimize geodesics under different Riemannian metrics on the toy datasets
├── requirements.txt
└── README.md
```
## Usage
```bash
git clone https://github.com/moritzburmester/riem-score-metrics.git
cd score-based-riemannian-metrics
conda create --name score-metrics --file requirements.txt
```
## Example Geodesics

### URC (Uniformly Rotated Characters)
<table>
  <tr>
    <td align="right" width="120"><b>ideal (γ*)</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/gamma_star_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>LERP</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/raw_lerp_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>SLERP</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/raw_slerp_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 0 (SAI)</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam0_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 0.1</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam01_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 0.25</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam025_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 0.5</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam05_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 0.75</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam075_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b> λ = 1</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/diff_lam1_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>EBM</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/Etheta_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>LAND</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/LAND_ex2.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>RBF</b></td>
    <td><img src="assets/urc_plots/geodesic_interpolations/RBF_ex2.png" width="850"></td>
  </tr>
</table>

### MorphBench (M) in Stable  v2.1-base — Dog → Cat
<table>
  <tr>
    <td align="right" width="120"><b>LERP</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lerp_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>SLERP</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_slerp_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0 (SAI)</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam00_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.1</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam01_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.25</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam025_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.5</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam05_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.75</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam075_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 1</b></td>
    <td><img src="assets/mb(m)_plots/geodesic_interpolations/mbm_realdog_cat_lam10_it500_strip.png" width="850"></td>
  </tr>
</table>
