%% stretch all images to 8bit
ratio = 1; % 1% linear stretch
maskp = 'mask.tif'; % the data path of mask, 1-vaild region, 0-invalid region
muxp = 'mux.tif'; % the data path of images

% load images
info = geotiffinfo(maskp);
[mask, R] = geotiffread(maskp);
[mux, ~] = geotiffread(muxp);

% stretch and save
mux8= func_16to8(mux, mask, ratio);
geotiffwrite('mux_quacs8.tif',mux8, R,...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);