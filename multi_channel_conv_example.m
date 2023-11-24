clearvars
clc

nRIR = 12;
c = 40;
r = 40;
l_1 = 8;
l_2 = 8;

nx = 100;
ny = r*c + nx - 1;

H = rand(r,c,nRIR);

[L,D,R] = GeneralizedLowRankApproximation(H,l_1,l_2,3);

W = zeros(r,l_2,nRIR);
H_hat = zeros(r,c,nRIR);
L_sym = L*L';
R_sym = R*R';

for j = 1:nRIR
    W(:,:,j) = L*D(:,:,j);
    H_hat(:,:,j) = L_sym*H(:,:,j)*R_sym;
end

x = rand(nx,1);

conv_multichannel = multiChannel_conv(W,R,x);
conv_matlab = zeros(ny,nRIR);

for j = 1:nRIR
    h = H_hat(:,:,j);
    h = h(:);
    conv_matlab(:,j) = conv(h,x);
end

max(max(abs(conv_matlab-conv_multichannel))) %Approximately 0 if all is well