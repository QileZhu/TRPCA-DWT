function M=convertVector2Mat(V,m,n)
M=zeros(m,n);
for i = 1:m
    M(i,:) = V(1+(i-1)*n:i*n)';
end