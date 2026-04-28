function weights = WDTNN_subband_adaptive_weights(X, W, l, epsilon, w_cap)
%WDTNN_SUBBAND_WEIGHTS  Subband-energy adaptive weights for WDTNN.
%
%   weights = WDTNN_subband_weights(X, W, l)
%   weights = WDTNN_subband_weights(X, W, l, epsilon, w_cap)
%
%   Inputs:
%       X        - observed tensor of size n1 x n2 x n3
%       W        - mode-3 DWT matrix of size n3 x n3
%       l        - DWT decomposition level
%       epsilon  - small constant to avoid division by zero, default 1e-16
%       w_cap    - upper bound for weights, default 1e8
%
%   Output:
%       weights  - n3 x 1 column vector of subband weights
%
%   The subband energy is computed by tensor Frobenius norm:
%       E_j = sqrt( sum_{i in I_j} ||Xbar(:,:,i)||_F^2 )
%
%   The weight for all slices in subband I_j is:
%       w_i = min( (max_q E_q + epsilon)/(E_j + epsilon), w_cap ),
%       i in I_j.

    if nargin < 4 || isempty(epsilon)
        epsilon = 1e-16;
    end
    if nargin < 5 || isempty(w_cap)
        w_cap = 1e8;
    end

    % Basic checks
    if ndims(X) ~= 3
        error('Input X must be a 3-D tensor.');
    end

    [~, ~, n3] = size(X);

    if size(W,1) ~= n3 || size(W,2) ~= n3
        error('W must be an n3 x n3 matrix, where n3 = size(X,3).');
    end

    if l < 1
        error('The decomposition level l must be at least 1.');
    end

    % Mode-3 DWT transform
    Xbar = dwt_mode3_mat(X, W);

    % Get DWT subband index sets
    subband_idx = local_dwt_subband_indices(n3, l);

    num_subbands = numel(subband_idx);
    E = zeros(num_subbands, 1);

    % Compute tensor Frobenius energy of each subband
    for j = 1:num_subbands
        idx = subband_idx{j};
        Xj = Xbar(:,:,idx);
        E(j) = sqrt(sum(Xj(:).^2));
    end

    Emax = max(E);

    % Compute subband weights and assign to all slices in each subband
    weights = zeros(n3, 1);

    for j = 1:num_subbands
        idx = subband_idx{j};

        wj = (Emax + epsilon) / (E(j) + epsilon);
        wj = min(wj, w_cap);

        weights(idx) = wj;
    end

    % Ensure column vector
    weights = weights(:);
end


function subband_idx = local_dwt_subband_indices(n3, l)
%LOCAL_DWT_SUBBAND_INDICES  Index sets for l-level 1-D DWT coefficients.
%
%   This assumes the common coefficient ordering:
%       [A_l, D_l, D_{l-1}, ..., D_1]
%
%   where length(A_l) = n3 / 2^l,
%         length(D_j) = n3 / 2^j.
%
%   Example for n3 = 256, l = 3:
%       A3: 1:32
%       D3: 33:64
%       D2: 65:128
%       D1: 129:256

    if mod(n3, 2^l) ~= 0
        error('n3 must be divisible by 2^l for this subband indexing rule.');
    end

    subband_idx = cell(l + 1, 1);

    % Approximation subband A_l
    len_A = n3 / 2^l;
    start_pos = 1;
    end_pos = len_A;
    subband_idx{1} = start_pos:end_pos;

    % Detail subbands: D_l, D_{l-1}, ..., D_1
    start_pos = end_pos + 1;

    for lev = l:-1:1
        len_D = n3 / 2^lev;
        end_pos = start_pos + len_D - 1;
        subband_idx{l - lev + 2} = start_pos:end_pos;
        start_pos = end_pos + 1;
    end

    if start_pos ~= n3 + 1
        error('Subband index construction failed. Please check n3 and l.');
    end
end