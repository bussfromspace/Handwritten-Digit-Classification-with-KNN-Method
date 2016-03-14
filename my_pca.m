function [Z, D_sort] = my_pca(X,k)

N = size(X,1);

m_x = mean(X);

Y = X - repmat(m_x,[N,1]); % center the data

S = 1/N * Y'*Y;            % covariance matrix

[C,D]  = eig(S);           % eigenvectors (C) and a diagonal matrix 
                           % containing eigenvalues (D)

[D_sort, id] = sort(diag(D), 'descend');

Z = zeros(N,k);
for i = 1:k
    Z(:,i) = Y * C(:,id(i));  
end
