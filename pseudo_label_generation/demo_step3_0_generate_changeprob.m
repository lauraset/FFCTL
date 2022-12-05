%% Function: generate change probability map 
ipath = 'D:\change\shanghai\';
% ipath = 'D:\change\beijing\';
%% load prediction probability map
p1 = [ipath, 'pred\', 'img18_update_scratch.tif'];
p2 = [ipath, 'pred\','img28_update_scratch.tif'];
respath = [ipath, 'diff'];

img1 = imread(p1);
img2 = imread(p2);

img1 = single(img1)/65535;
img2 = single(img2)/65535;

diff = abs(img1-img2);

[~,R] = geotiffread(p1);
info = geotiffinfo(p1);

% save
% 8bit
% max(diff(:));
diff8 = uint8(diff*255);
resname= fullfile(respath,'diffprob8.tif');
geotiffwrite(resname, diff8, R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
