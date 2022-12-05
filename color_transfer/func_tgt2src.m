function res = func_tgt2src(Im_target,Im_source)
% bgr
Im_trg_d=  im2double(Im_target(:,:,:));%1:3));
Im_src_d = im2double(Im_source(:,:,:));%1:3));
res = Color_Transfer_CCS_multi(Im_trg_d,Im_src_d);
% gr-nir
% Im_trg_d=  im2double(Im_target(:,:,2:4));
% Im_src_d = im2double(Im_source(:,:,2:4));
% IR2= Color_Transfer_CCS_multi(Im_trg_d,Im_src_d);
% Im_trg_d=  im2double(padarray(Im_target(:,:,4),[0,0,1],'replicate'));
% Im_src_d = im2double(padarray(Im_source(:,:,4),[0,0,1],'replicate'));
% IR2= Color_Transfer_CCS_multi(Im_trg_d,Im_src_d);

% merge
% res = zeros(size(Im_target));
% res(:,:,1:3) = IR1;
% res(:,:,4) = Im_target(:,:,4); % last dimension
end