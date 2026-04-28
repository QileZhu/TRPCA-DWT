function X = prox_trpca_dwt(Y,rho,W,weight)

[n1,n2,n3] = size(Y);
X_dwt = zeros(n1,n2,n3);

Y_dwt = dwt_mode3(Y, W);

w_rho=weight*rho;
for i = 1 : n3
    [U,S,V] = svd(Y_dwt(:,:,i),'econ');
    S = diag(S);
    r = length(find(S>w_rho(i)));
    S = S(1:r)-rho;
    X_dwt(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
end

X = idwt_mode3(X_dwt, W);
