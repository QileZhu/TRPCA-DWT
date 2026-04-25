function Y = dwt_mode3_mat(X, W)
% X: n1×n2×n3
% W: n3×n3

[n1,n2,n3] = size(X);

X_mat = reshape(X, [], n3);  
Y_mat = X_mat * W';         
Y = reshape(Y_mat, n1, n2, n3);
end