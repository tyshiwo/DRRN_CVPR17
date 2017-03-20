clear;close all;
setenv('LC_ALL','C');

%% settings
folder = './Train_291/';
size_input = 31;
size_label = 31;
stride = 21;
savepath = ['./train_291_' num2str(size_input) '_x234.h5'];

% scale augmentation
scale_2 = 2; 
scale_3 = 3; 
scale_4 = 4; 

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label,1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    disp(['i = ' num2str(i)]);
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image); 
    image_luminance = image(:,:,1); % lumination
    image = im2double(image_luminance);
    
    % scale 2
    im_label_2 = modcrop(image, scale_2);
    [hei_2,wid_2] = size(im_label_2);
    im_input_2 = imresize(imresize(im_label_2,1/scale_2,'bicubic'),[hei_2,wid_2],'bicubic');
%     imshow([im_label_2 im_input_2],[]);

    for x = 1 : stride : hei_2-size_input+1
        for y = 1 :stride : wid_2-size_input+1
            
            subim_input = im_input_2(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label_2(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            
            count=count+1;
            data_temp = subim_input;
            label_temp = subim_label;
            
            data(:, :, :, count) = single(data_temp);
            label(:, :, :, count) = single(label_temp);
        end
    end
    
    % scale 3
    im_label_3 = modcrop(image, scale_3);
    [hei_3,wid_3] = size(im_label_3);
    im_input_3 = imresize(imresize(im_label_3,1/scale_3,'bicubic'),[hei_3,wid_3],'bicubic');
%     imshow([im_label_3 im_input_3],[]);

    for x = 1 : stride : hei_3-size_input+1
        for y = 1 :stride : wid_3-size_input+1
            
            subim_input = im_input_3(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label_3(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            
            count=count+1;
            data_temp = subim_input;
            label_temp = subim_label;
            
            data(:, :, :, count) = single(data_temp);
            label(:, :, :, count) = single(label_temp);
        end
    end
    
    % scale 4
    im_label_4 = modcrop(image, scale_4);
    [hei_4,wid_4] = size(im_label_4);
    im_input_4 = imresize(imresize(im_label_4,1/scale_4,'bicubic'),[hei_4,wid_4],'bicubic');
%     imshow([im_label_4 im_input_4],[]);

    for x = 1 : stride : hei_4-size_input+1
        for y = 1 :stride : wid_4-size_input+1
            
            subim_input = im_input_4(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label_4(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            
            count=count+1;
            data_temp = subim_input;
            label_temp = subim_label;
            
            data(:, :, :, count) = single(data_temp);
            label(:, :, :, count) = single(label_temp);
        end
    end
    
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
