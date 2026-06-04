function M = convertVector2Mat(V,m,n)
%CONVERTVECTOR2MAT  Convert a row-wise vectorized image back to a matrix.
%
%   M = convertVector2Mat(V,m,n) reshapes the vector V into an m-by-n image
%   matrix. This function is the inverse of convertMat2Vector.m under the
%   row-wise vectorization convention used in the background modeling demo.
%
%   Inputs:
%       V - vector of length m*n
%       m - number of image rows
%       n - number of image columns
%
%   Output:
%       M - reconstructed m-by-n matrix

M = zeros(m,n);
for i = 1:m
    M(i,:) = V(1+(i-1)*n:i*n)';
end
end
