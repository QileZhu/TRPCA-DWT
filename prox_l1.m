function x = prox_l1(b,lambda)
%PROX_L1  Proximal operator of lambda*||x||_1.
%
%   x = prox_l1(b,lambda) applies element-wise soft-thresholding:
%       x = sign(b).*max(abs(b)-lambda,0).
%
%   Inputs:
%       b      - input array
%       lambda - nonnegative threshold
%
%   Output:
%       x      - thresholded array

x = max(0,b-lambda) + min(0,b+lambda);
end
