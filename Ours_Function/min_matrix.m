function [min_mat, result] = min_matrix(I, patch_size)
    
    I = min(I,[],3); [M, N] = size(I);
    J_index = zeros(M, N); % Create empty index matrix
    % Test if patch size has odd number
    if ~mod(numel(patch_size),2) % if even number
        error('Invalid Patch Size: Only odd number sized patch supported.');
    end
    padsize = floor(patch_size/2);

    for m = 1:M
        for n = 1:N
            ind_row1 = max(1,m-padsize); ind_row2 = min(M,m+padsize);
            ind_col1 = max(1,n-padsize); ind_col2 = min(N,n+padsize);
            patch = I(ind_row1:ind_row2,ind_col1:ind_col2);
            [m2,loc] = location(patch);
            real_row = ind_row1 + loc(1) - 1;
            real_col = ind_col1 + loc(2) - 1;
            J_index(m,n) = (real_col - 1) * M + real_row;
        end
    end
    sparse_index_row = (1:M*N)';
    ss = ones(M,N);
    ss = ss(:);
    sparse_index_col = J_index(:);
    min_mat = sparse(sparse_index_row, sparse_index_col,ss, M*N, M*N, M*N);
    result = min_mat*I(:);
end





