## DR-EnKF
This repository contains the source code implementation of the paper: "Deep learning-enhanced reduced-order ensemble Kalman filter for efficient Bayesian data assimilation of parameteric PDEs".  To run the experiment, you just need to run the file "DR_EnKF_main.py". For other experiments, the code is similiar, you can implement it easily.
 ## Abstract 
 Bayesian data assimilation for systems governed by parametric partial differential equations (PDEs) is computationally demanding due to the need for multiple forward model evaluations. Reduced-order models (ROMs) have been widely used to reduce the computational burden. However, traditional ROM techniques rely on linear mode superposition, which frequently fails to capture nonlinear time-dependent dynamics efficiently and leads to biases in the assimilation results. To address these limitations, we introduce a new deep learning-enhanced reduced-order ensemble Kalman filter (DR-EnKF) method for Bayesian data assimilation. The proposed approach employs a two-tiered learning framework. First, the full-order model is reduced using operator inference, which finds the primary dynamics of the system through long-term simulations generated from coarse-grid data. Second, a model error network is trained with short-term simulation data from a fine grid to learn about the ROM-induced discrepancy. The learned network is then used online to correct the ROM-based EnKF, resulting in more accurate state updates during the assimilation process. The performance of the proposed method is evaluated on several benchmark problems, including the Burgers’ equation, the FitzHugh-Nagumo model, and advection-diffusion-reaction systems. The results show considerable computational speedup without compromising accuracy, making this approach an effective tool for large-scale data assimilation tasks.
 ## Citation

```
@article{WANG2025109544,
title = {Deep learning-enhanced reduced-order ensemble Kalman filter for efficient Bayesian data assimilation of parametric PDEs},
journal = {Computer Physics Communications},
volume = {311},
pages = {109544},
year = {2025},
doi = {https://doi.org/10.1016/j.cpc.2025.109544},
author = {Yanyan Wang and Liang Yan and Tao Zhou}
}
```
