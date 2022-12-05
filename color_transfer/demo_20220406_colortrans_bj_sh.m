%% color transfer on beijing 

% ipath = 'D:\change\data_bj\';
ipath = 'D:\change\data_sh\';
imglist1 = dir([ipath, 'img1\', '*.tif']);
imglist2 = dir([ipath, 'img2\', '*.tif']);

res1 = [ipath,'img1t'];
res2 = [ipath,'img2t'];
if ~isfolder(res1)
    mkdir(res1);
end
if ~isfolder(res2)
    mkdir(res2);
end

tic;
num = length(imglist1);
parfor i=1:num
    n1  =  imglist1(i).name;
    n2  = imglist2(i).name;
    f1 = fullfile(imglist1(i).folder, n1);
    f2 = fullfile(imglist2(i).folder, n2);
    im1 = imread(f1);
    im2 = imread(f2);
    % im1 -> im2
    im1t = func_tgt2src(im1,im2);
    % im2 -> im1
%     im2t = func_tgt2src(im2,im1);
    % save
    im1t = uint8(im1t*255);
    tiffwrite(im1t, fullfile(res1, n1));
%     im2t = uint8(im2t*255);
%     tiffwrite(im2t, fullfile(res2, n2));
end 
toc;
%     figure(1);
%     subplot(1,4,1);
%     imshow(im1(:,:,3:-1:1));
%     subplot(1,4,3);
%     imshow(im2(:,:,3:-1:1));
%     subplot(1,4,2);
%     imshow(im1t(:,:,3:-1:1));
%     subplot(1,4,4);
%     imshow(im2t(:,:,3:-1:1));  
