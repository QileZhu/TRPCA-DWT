function [L,S,iter] = TRPCA_DWT(X,level, W, weight)
[n1,n2,n3] = size(X);
if nargin < 3
    % level=1;
    wname='haar';
    weight=ones(n3,1);
    W = dwt_matrix(n3, level, wname);
end

tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;

lambda = 1/sqrt(max(n1,n2));
dim = size(X);
L = zeros(dim);
S = L;
Y = L;

for iter = 1 : max_iter
    Lk = L;
    Sk = S;
    % update L
    L = prox_trpca_dwt(-S+X-Y/mu,1/mu,W,weight);

    % update S
    S = prox_l1(-L+X-Y/mu,lambda/mu);

    dY = L+S-X;
    chgL = max(abs(Lk(:)-L(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([ chgL chgS max(abs(dY(:))) ]);
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);
end

