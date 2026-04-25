function X = idwt_mode3_mat(Y, W)

[n1,n2,n3] = size(Y);

Y_mat = reshape(Y, [], n3);

X_mat = Y_mat * W;   

X = reshape(X_mat, n1, n2, n3);
end