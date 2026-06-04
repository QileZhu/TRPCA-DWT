function X = idwt_mode3(Y, W)
%IDWT_MODE3  Apply the inverse DWT matrix along the third mode of a tensor.
%
%   X = idwt_mode3(Y, W) transforms the DWT-domain tensor Y back to the
%   original domain. For an orthogonal DWT matrix W, the inverse operation
%   of Y_mat = X_mat * W' is X_mat = Y_mat * W.
%
%   Inputs:
%       Y - transformed tensor of size n1 x n2 x n3
%       W - orthogonal DWT matrix of size n3 x n3
%
%   Output:
%       X - reconstructed tensor of size n1 x n2 x n3

[n1,n2,n3] = size(Y);

% Reshape mode-3 tubes into rows, apply inverse transform, and reshape back.
Y_mat = reshape(Y, [], n3);
X_mat = Y_mat * W;
X = reshape(X_mat, n1, n2, n3);
end
