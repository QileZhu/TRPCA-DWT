# TRPCA-DWT

This repository provides MATLAB code for **DWT-based Tensor Robust Principal Component Analysis (TRPCA-DWT)**.
The method decomposes an observed third-order tensor `X` into a low-rank component `L` and a sparse component `S` by using a mode-3 Discrete Wavelet Transform (DWT) and an adaptive subband-weighted DWT-based tensor nuclear norm (WDTNN).

## Files

- `main_background_model.m`: demo script for video background modeling.
- `TRPCA_DWT.m`: main ADMM solver for TRPCA-DWT.
- `prox_trpca_dwt.m`: weighted DWT-domain singular value thresholding operator.
- `WDTNN_subband_adaptive_weights.m`: adaptive subband-wise weight computation.
- `dwt_matrix.m`: construction of the DWT transform matrix.
- `dwt_mode3.m` / `idwt_mode3.m`: mode-3 DWT and inverse mode-3 DWT.
- `prox_l1.m`: soft-thresholding operator for the sparse component.
- `convertMat2Vector.m` / `convertVector2Mat.m`: helpers for vectorizing and reconstructing video frames.
- `Dataset/`: sample `.mat` files used by the demo.

## Requirements

- MATLAB R2020a or later is recommended.
- MATLAB Wavelet Toolbox is required for `wavedec` in `dwt_matrix.m`.

## Quick start

1. Open MATLAB and set the current folder to the root of this repository.
2. Run:

```matlab
main_background_model
```

The script loads `Dataset/HighwayI.mat`, runs TRPCA-DWT, and saves the output frames to:

```text
Output/HighwayI/L   % recovered low-rank background frames
Output/HighwayI/S   % recovered sparse foreground frames
```

To run another included dataset, edit `main_background_model.m` and switch the `load` command from `HighwayI.mat` to `IBMtest2.mat`.

## Basic function call

```matlab
[L,S,iter] = TRPCA_DWT(X);
```

Optional arguments can specify the DWT level and wavelet basis:

```matlab
[L,S,iter] = TRPCA_DWT(X, level, 'haar');
```

where `X` is an `n1 x n2 x n3` tensor and DWT is applied along the third mode.

## Notes for reproducing experiments

The demo provides a runnable example for background modeling. For large-scale experiments reported in the paper, prepare the corresponding datasets in the same tensor format and call `TRPCA_DWT` with the same preprocessing settings described in the manuscript.
