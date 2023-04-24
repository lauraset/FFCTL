function res = func_16to8(im, mask, ratio)
im=single(im);
[r,c,b]=size(im);
im_=reshape(im,[r*c,b]);
v=prctile(im_(mask(:)~=0,:),[ratio,100-ratio],1); 
res=im;
for i=1:b
    data=im(:,:,i);
    tmp=(data-v(1,i))./(v(2,i)-v(1,i)+eps);
    tmp(data<=v(1,i))=0;tmp(data>=v(2,i))=1;
    res(:,:,i)=tmp;
end
% to 8bit
res = uint8(res*255);
end