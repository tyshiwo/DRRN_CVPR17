function PSNR_B = compute_psnrb(im_gnd, im)

if size(im_gnd, 3) == 3,
    im_gnd = rgb2ycbcr(im_gnd);
    im_gnd = im_gnd(:, :, 1);
end

if size(im, 3) == 3,
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end

imdff = double(im_gnd) - double(im);
imdff = imdff(:);

MSE = mean(imdff.^2);
BEF = compute_bef(im);
MSE_B = MSE + BEF;

if max(im(:))>2
    PSNR_B = 10 * log10(255^2 / MSE_B);
else
    PSNR_B = 10 * log10(1 / MSE_B);
end