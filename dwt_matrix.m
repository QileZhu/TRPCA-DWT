function W = dwt_matrix(n, level, wname)

I = eye(n);

W = zeros(n,n);

for i = 1:n
    e = I(:,i)';      
    [c,~] = wavedec(e, level, wname);
    
    W(:,i) = c(:);
end
end