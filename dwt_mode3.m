function Y = dwt_mode3(X, W)
%DWT_MODE3  Apply a DWT matrix along the third mode of a tensor.
%
%   Y = dwt_mode3(X, W) computes the mode-3 DWT-domain tensor Y from X.
%   If X is n1 x n2 x n3 and W is n3 x n3, then each mode-3 tube of X is
%   transformed by W.
%
%   Inputs:
%       X - input tensor of size n1 x n2 x n3
%       W - DWT matrix of size n3 x n3
%
%   Output:
%       Y - transformed tensor of size n1 x n2 x n3

[n1,n2,n3] = size(X);

% Reshape all mode-3 tubes into rows, apply the transform, and reshape back.
X_mat = reshape(X, [], n3);
Y_mat = X_mat * W';
Y = reshape(Y_mat, n1, n2, n3);
end
