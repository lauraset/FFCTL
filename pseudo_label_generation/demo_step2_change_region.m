%% Function: Uncertainty-aware pseudo label generation
% The algorithm includes three steps: 
% 1) single-temporal building prediction
% 2) object-to-pixel multi-temporal comparison; 
% 3) uncertainty-aware analysis for reliable pseudo label generation.
% Author: Yinxia Cao
%% define data path
citypath = 'D:\change\beijing\';
% citypath = 'D:\change\shanghai\';
%% load building prediction results.
p1 = fullfile(citypath, 'pred','img18_update_scratch_seg.tif');
p2 = fullfile(citypath, 'pred','img28_update_scratch_seg.tif');
respath = fullfile(citypath, 'diff');
if ~isfolder(respath)
    mkdir(respath);
end

im1 = imread(p1);
im1(im1==255)=1;% imshow(im1,[]);

im2 = imread(p2);
im2(im2==255)=1;% imshow(im2,[]);

[~,R] = geotiffread(p1);
info = geotiffinfo(p1);

%% find changed pixels
im1s = single(im1);
im2s = single(im2);
diffpixel = (im2s-im1s);
% imshow(diffpixel,[]);
diffpixel(diffpixel>0)=2; % 1->255,positive 
diffpixel(diffpixel<0)=1; % -1 -> 128negative
resname= fullfile(respath,'diffpix.tif');
geotiffwrite(resname, uint8(diffpixel), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 0. overlay analysis
area_thr = 20; % 125 m2
union = (im1|im2);% imshow(union);
cc = bwconncomp(union);
%% 1. area filter
stats=regionprops(cc,'Area');
idx=find([stats.Area]>=area_thr);
union_f1=ismember(labelmatrix(cc),idx);
% save
resname= fullfile(respath,'union_area.tif');
geotiffwrite(resname, uint8(union_f1), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 2.calculate IOU for each region
cc = bwconncomp(union_f1);
num = cc.NumObjects;
numover = zeros(num,4); % produce the number of overlay pixels
for i=1:num
    pos = cc.PixelIdxList{i};
    t1 = im1(pos);
    t2 = im2(pos);
    numover(i,1) = sum(t1);
    numover(i,2) = sum(t2);
    numover(i,3) = sum(t1&t2); % inter
    numover(i,4) = length(pos); % union   
end
iou = numover(:,3)./numover(:,4);
% histogram(iou);
% xlabel('IOU');
% ylabel('#Pixels');
save(fullfile(respath,'union_iou.mat'),'iou');
%% 3.changed objects: IOU < 0.5
tiou = 0.5; %0.5;
nummax = max(numover(:,1:2),[], 2); % area filter
idchange = find((iou<tiou) & (nummax>=area_thr));
union_change=ismember(labelmatrix(cc),idchange);
imshow(union_change);
% save
resname= fullfile(respath,'union_change.tif');
geotiffwrite(resname, uint8(union_change), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 4. unchanged objects: IOU>=0.5
idunchange =  find((iou>=tiou) & (nummax>=area_thr)); % area filter
union_unchange=ismember(labelmatrix(cc),idunchange);% imshow(union_unchange);
% save
resname= fullfile(respath,'union_unchange.tif');
geotiffwrite(resname, uint8(union_unchange), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 5. merge changed and unchanged objects
res = zeros(size(union_change),'uint8');
res(union_unchange==1)=1; % unchange
res(union_change==1)=2; % change
resname= fullfile(respath,'union_all.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
% color
res(res==1)=128; % unchange
res(res==2)=255; % change
resname= fullfile(respath,'union_allc.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 6. remove changed pixel from changed objects
union_all = imread(fullfile(respath,'union_all.tif'));
union_all_diff = union_all;
union_all_diff((diffpixel==0) & (union_all==2)) = 1; % unchanged pixels in changed objects
resname= fullfile(respath,'union_alldiff.tif');
geotiffwrite(resname, uint8(union_all_diff), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

