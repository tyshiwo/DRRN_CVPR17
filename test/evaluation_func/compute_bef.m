function BEF = compute_bef(im)

if nargin < 1
    im=imread('matlab_temporary_jpeg_SRCNN.jpg');
end

B = 8;

[hei, wid, dep] = size(im);
if dep > 1
    disp('Not for color images');
end

H = 1 : (wid - 1);
H_b = B : B : (wid - 1);
H_bc = setxor(H, H_b);
V = 1 : (hei - 1);
V_b = B : B : (hei - 1);
V_bc = setxor(V, V_b);

D_b = 0;
D_bc = 0;

for i = H_b
    diff = im(: , i) - im(: , i + 1);
    D_b = D_b + sum(diff.^2);
end

for i = H_bc
    diff = im(: , i) - im(: , i + 1);
    D_bc = D_bc + sum(diff.^2);
end

for j = V_b
    diff = im(j , :) - im(j + 1 , :);
    D_b = D_b + sum(diff.^2);
end

for j = V_bc
    diff = im(j , :) - im(j + 1 , :);
    D_bc = D_bc + sum(diff.^2);
end

N_hb = hei * (wid/B - 1);
N_hbc = hei * (wid - 1) - N_hb;
N_vb = wid * (hei/B - 1);
N_vbc = wid * (hei - 1) - N_vb;

D_b = D_b / (N_hb + N_vb);
D_bc = D_bc / (N_hbc + N_vbc);

if D_b > D_bc
    T = log2(B) / log2(min(hei, wid));
else
    T = 0;%log2(B) / log2(min(hei, wid));
end

BEF = T * (D_b - D_bc);