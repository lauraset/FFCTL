%% Function: select reliable labels 
% 1) Obtain all buildings; 2) then, obtain change, unchange, uncertain reigons
ipath = 'D:\change\shanghai\';
% ipath = 'D:\change\beijing\';
respath = [ipath, 'diff'];

probp = fullfile(respath,'diffprob8.tif');
[prob, R] = geotiffread(probp);
info= geotiffinfo(probp);
prob = single(prob)/255.0;

labp = fullfile(respath, 'union_alldiff.tif');
lab = imread(labp);

imshow(lab, []);

t1 = 0.7;
t2 = 1-t1;
% 1-cert, 0-uncert
cert_change = uint8(prob>t1); % change 
cert_unchange = uint8(prob<t2); % unchange

%% set the value of uncertain region to 3
labnew = lab;
labnew((lab==2) & (cert_change==0)) = 3; % uncert changed region 
labnew((lab==1) & (cert_unchange==0)) = 3; % uncert unchanged region 
labnew((lab==0) & (cert_unchange==0)) = 3; % uncert unchanged region 

resname= fullfile(respath,'union_alldiff_cert.tif');
geotiffwrite(resname, labnew, R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

