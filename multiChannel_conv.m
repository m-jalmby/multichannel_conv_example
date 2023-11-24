function y = multiChannel_conv(W,R,x)
%MULTICHANNEL_CONV Mutli-channel conv
r = size(W,1);
l2 = size(W,2);
c = size(R,1);

if size(W,2) ~= size(R,2)
    error('Wrong size')
end

nx = length(x);
nh = r*c;
ny = nh + nx - 1;
nbrRIRs = size(W,3);
y = zeros(ny,nbrRIRs);
P_tilde = zeros(nx+(c-1)*r,l2);
P = (1:1:length(P_tilde))';
y_vec = (1:1:ny)';

R_low = max(ceil((P-nx)/r)+1,1);
R_high = min(ceil(P/r),c);
x_low = max(mod(P-1,r)+1,P-(c-1)*r);
x_high = min(nx-r+1+mod(P-1-nx,r),P);
w_start = max(1,y_vec-(ny-r));
w_stop = min(y_vec,r);
P_start = max(y_vec-r+1,1);
P_stop = min(y_vec,ny-r+1);

%Creating P
for j1 = 1:size(P_tilde,1)
    for j2 = 1:l2
        R_j = R(R_low(j1):R_high(j1),j2);
        x_j = flip(x(x_low(j1):r:x_high(j1)));
        P_tilde(j1,j2) = dot(R_j,x_j);
    end
end

%Create y
for j1 = 1:nbrRIRs
    for j2 = 1:ny
        for j3 = 1:l2
            w_vec = flip(W(w_start(j2):w_stop(j2),j3,j1));
            P_vec = P_tilde(P_start(j2):P_stop(j2),j3);
            y(j2,j1) = y(j2,j1) + dot(w_vec,P_vec);
        end
    end
end
end