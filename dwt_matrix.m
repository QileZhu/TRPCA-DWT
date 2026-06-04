function W = dwt_matrix(n, level, wname)
%DWT_MATRIX  Construct the matrix representation of an l-level 1-D DWT.
%
%   W = dwt_matrix(n, level, wname) returns an n-by-n matrix W such that
%   x_bar = x * W' gives the wavelet coefficients of a row vector x under
%   the same coefficient ordering as MATLAB wavedec.
%
%   Inputs:
%       n      - signal length along the transform mode
%       level  - DWT decomposition level
%       wname  - wavelet basis name, e.g., 'haar', 'db2', 'sym2'
%
%   Output:
%       W      - DWT transform matrix
%
%   Requirement:
%       This function requires MATLAB Wavelet Toolbox for wavedec.

I = eye(n);
W = zeros(n,n);

% Apply DWT to every canonical basis vector to obtain the transform matrix.
for i = 1:n
    e = I(:,i)';
    [c,~] = wavedec(e, level, wname);
    W(:,i) = c(:);
end
end
