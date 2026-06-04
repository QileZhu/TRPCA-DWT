function [L,S,iter] = TRPCA_DWT(X,level,wname)
%TRPCA_DWT  Tensor Robust PCA with mode-3 DWT and adaptive WDTNN.
%
%   [L,S,iter] = TRPCA_DWT(X)
%   [L,S,iter] = TRPCA_DWT(X,level,wname)
%
%   Solves the TRPCA-DWT model
%       min_{L,S} ||L||_{WDTNN} + lambda*||S||_1,  s.t. X = L + S,
%   using an ADMM scheme. The low-rank tensor L is regularized by the
%   adaptive subband-weighted DWT-based tensor nuclear norm (WDTNN), and S
%   is the sparse corruption/foreground component.
%
%   Inputs:
%       X      - observed third-order tensor of size n1 x n2 x n3
%       level  - DWT decomposition level. Default: log2(n3)
%       wname  - wavelet basis name. Default: 'haar'
%
%   Outputs:
%       L      - recovered low-rank tensor
%       S      - recovered sparse tensor
%       iter   - number of ADMM iterations
%
%   Notes:
%       The third tensor mode is the transform mode along which DWT is
%       applied. In video background modeling, the tensor is usually
%       arranged so that the third mode corresponds to frames/time.

[n1,n2,n3] = size(X);

% Default settings used in the paper.
if nargin < 2 || isempty(level)
    level = log2(n3);
end
if nargin < 3 || isempty(wname)
    wname = 'haar';
end

% Construct the orthogonal mode-3 DWT matrix and adaptive subband weights.
Mdwt = dwt_matrix(n3, level, wname);
weight = WDTNN_subband_adaptive_weights(X, Mdwt, level);

% ADMM parameters.
tol = 1e-8;
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;

% Standard TRPCA sparsity parameter.
lambda = 1/sqrt(max(n1,n2));

% Initialize variables.
dim = size(X);
L = zeros(dim);
S = L;
Y = L;

for iter = 1 : max_iter
    Lk = L;
    Sk = S;

    % Update low-rank component by weighted DWT-domain singular value
    % thresholding. The threshold is weight(i)/mu for the i-th transformed
    % frontal slice.
    L = prox_trpca_dwt(-S+X-Y/mu, 1/mu, Mdwt, weight);

    % Update sparse component by element-wise soft-thresholding.
    S = prox_l1(-L+X-Y/mu, lambda/mu);

    % Primal residual and stopping criterion.
    dY = L + S - X;
    chgL = max(abs(Lk(:)-L(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([chgL, chgS, max(abs(dY(:)))]);

    if chg < tol
        break;
    end

    % Update multiplier and penalty parameter.
    Y = Y + mu*dY;
    mu = min(rho*mu, max_mu);
end
