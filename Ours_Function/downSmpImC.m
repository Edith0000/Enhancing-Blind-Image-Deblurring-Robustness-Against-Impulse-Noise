function sI=downSmpImC(I,ret) %对图像进行下采样，ret（比例因子）
%% refer to Levin's code
if (ret==1) 
    sI=I;
    return % 直接原图输出
end
%%%%%%%%%%%%%%%%%%%

sig=1/pi*ret; % 根据比例因子ret计算高斯滤波器的标准差sig

g0=(-50:50)*2*pi;
sf=exp(-0.5*g0.^2*sig^2); % 构建一个高斯滤波器，核大小为[-50:50]
sf=sf/sum(sf); % 归一化
csf=cumsum(sf); % cumsum函数（计算各行的累加和）
csf=min(csf,csf(end:-1:1));
ii=csf>0.05;

sf=sf(ii); % 根据索引值ii从sf中提取对应的元素，得到新的高斯滤波器核sf
sum(sf); 

I=conv2(sf,sf',I,'valid'); % 对图像I进行卷积操作

[gx,gy]=meshgrid(1:1/ret:size(I,2),1:1/ret:size(I,1)); %构建采样网格gx，gy

sI=interp2(I,gx,gy,'bilinear'); % 运用双线性插值的方法，对图像进行插值操作
end