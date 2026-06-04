function X = prox_trpca_dwt(Y,rho,W,weight)
%PROX_TRPCA_DWT  Proximal operator of the weighted DWT tensor nuclear norm.
%
%   X = prox_trpca_dwt(Y,rho,W,weight) computes
%       argmin_X  ||X||_{WDTNN} + (1/(2*rho))*||X-Y||_F^2
%   by applying mode-3 DWT, performing weighted singular value thresholding
%   on each transformed frontal slice, and applying inverse mode-3 DWT.
%
%   Inputs:
%       Y      - input tensor
%       rho    - proximal threshold scale, usually 1/mu in ADMM
%       W      - orthogonal DWT matrix along mode-3
%       weight - n3 x 1 vector of WDTNN slice weights
%
%   Output:
%       X      - result after weighted DWT-domain SVT

[n1,n2,n3] = size(Y);
X_dwt = zeros(n1,n2,n3);

% Transform input tensor to the DWT domain.
Y_dwt = dwt_mode3(Y, W);

% Weighted threshold for each transformed frontal slice.
w_rho = weight(:) * rho;

for i = 1:n3
    % Economy SVD of the i-th transformed frontal slice.
    [U,S,V] = svd(Y_dwt(:,:,i),'econ');
    s = diag(S);

    % Keep singular values larger than the weighted threshold.
    r = nnz(s > w_rho(i));
    if r > 0
        s_shrink = s(1:r) - w_rho(i);
        X_dwt(:,:,i) = U(:,1:r) * diag(s_shrink) * V(:,1:r)';
    end
end

% Transform back to the original tensor domain.
X = idwt_mode3(X_dwt, W);
end
