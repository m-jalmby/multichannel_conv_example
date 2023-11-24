function [L,D,R,conversionProgression] = GeneralizedLowRankApproximation(H,l_1,l_2,maxIter,R,L)
%GENERALIZEDLOWRANKAPPROXIMATION Implements the algorithm from the paper
%Generalized Low Rank Approximation of Matrices, by Jieping Ye.
%Inputs:
%H - tensor where the slices are the matricezed RIRs
%l_1 - number of rows for D_i
%l_2 - number of cols for D_i
%maxIter - number of iterations (5 - 10 is plenty)
%L - optional, if already known and we wish to optimize wrt R
%R - optional, if already known and we wish to optimize wrt L
%Output:
%L - transformation from the left
%D - tensor where the slices are the reduced representations of H
%R - transformation from the right

if nargout>3
    conversionProgression = zeros(maxIter,1);
else
    conversionProgression = [];
end

if (nargin < 5) || (isempty(inputname(5)))
    R_unkown = 1;
else
    R_unkown = 0;
    maxIter = 1;
end

if (nargin < 6) || (isempty(inputname(6)))
    L_unkown = 1;
else
    L_unkown = 0;
    maxIter = 1;
end

H = squeeze(H); %Remove any dimension with size(1);

if length(size(H)) ~= 3
    if length(size(H)) == 2
        H = reshape(H,size(H,1),size(H,2),1);
    else
        error('Three-dimensional tensor, please')
    end
end

r = size(H,1);
n = size(H,3);

D = zeros(l_1,l_2,n);

L0 = [eye(l_1);zeros(r-l_1,l_1)]; %Initialization for L, in accorance with recommendation in paper
L_j_minus_1 = L0;
if R_unkown && ~L_unkown
    L_j_minus_1 = L;
end

for j = 1:maxIter
    if R_unkown
        M_R = zeros(size(H,2),size(H,2));
        for jj = 1:n
            M_R = M_R + H(:,:,jj)'*(L_j_minus_1*L_j_minus_1')*H(:,:,jj);
        end
        [phi,eig_val_mat] = eig(M_R);
        [~,ind] = sort(abs(diag(eig_val_mat)),'descend');
        phi = phi(:,ind);
        R_j = phi(:,1:l_2);
    else
        R_j = R;
    end

    if L_unkown
        M_L = zeros(size(H,1),size(H,1));
        for jj = 1:n
            M_L = M_L + H(:,:,jj)*(R_j*R_j')*H(:,:,jj)';
        end
        [phi,eig_val_mat] = eig(M_L);
        [~,ind] = sort(abs(diag(eig_val_mat)),'descend');
        phi = phi(:,ind);
        L_j = phi(:,1:l_1);
        L_j_minus_1 = L_j;
    else
        L_j = L;
    end

    if nargout>3
        for jj = 1:n
            H_approx = (L_j*L_j')*H(:,:,jj)*(R_j*R_j');
            conversionProgression(j) = conversionProgression(j) + norm(H(:,:,jj)-H_approx,'fro');
        end
    end
end

for jj = 1:n
    D(:,:,jj) = L_j'*H(:,:,jj)*R_j;
end

L = L_j;
R = R_j;
end

