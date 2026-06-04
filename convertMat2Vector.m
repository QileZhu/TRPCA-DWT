function vector = convertMat2Vector(M)
%CONVERTMAT2VECTOR  Convert an image matrix to a column vector.
%
%   vector = convertMat2Vector(M) reshapes a 2-D image matrix M into a
%   column vector using row-wise scanning. This is useful for constructing
%   a tensor whose first mode stacks spatial pixels.
%
%   The inverse operation is implemented in convertVector2Mat.m.

TM = M';
vector = TM(:);
end
