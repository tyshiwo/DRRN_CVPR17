clear;
addpath('../../CVPR16_VDSR/');
addpath('./matlabPyrTools-master/');
%% test
% load('DRRN_22C128_dataset_1_x3');
load('VDSR_20C64_dataset_1_x3');
path = ['../../SR Test dataset/Set5/'];
d = dir([path '*.bmp']);
up_scale = 3;
sum = 0 ;
for i=1:length(d)
    im = imread([path d(i).name]);
    %% work on illuminance only
    if size(im,3)>1
        im = rgb2ycbcr(im);
    end
    im = im2double(im(:,:,1));
    im_gnd = modcrop(im, up_scale);
    [hei,wid] = size(im_gnd);
    im_b = imresize(imresize(im_gnd,1/up_scale,'bicubic'),up_scale,'bicubic');
    %% remove border
    im_b1 = shave(uint8(single(im_b) * 255), [up_scale, up_scale]);
    im_gnd1 = shave(uint8(single(im_gnd) * 255), [up_scale, up_scale]);
    im_h1 = im_h_set{i};
    sum = sum+ifcvec(double(im_gnd1),double(im_h1));
end
ifc = sum/length(d)


 