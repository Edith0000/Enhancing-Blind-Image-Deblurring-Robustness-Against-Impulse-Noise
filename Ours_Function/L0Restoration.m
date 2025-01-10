function S = L0Restoration(Im, kernel, lambda, kappa)
if ~exist('kappa','var')
    kappa = 2.0;
end
%% pad image
H = size(Im,1);    W = size(Im,2);
Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1)); % 扩充图像边缘
%%
S = Im;
betamax = 1e5;
fx = [1, -1]; % 水平方向导数滤波器
fy = [1; -1]; % 垂直方向导数滤波器
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D); % 将滤波器转化到频域
otfFy = psf2otf(fy,sizeI2D);
%%
KER = psf2otf(kernel,sizeI2D); % 将kernel转化到频域
Den_KER = abs(KER).^2; % 计算KER模的平方
%%
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1% 扩展通道数
    Denormin2 = repmat(Denormin2,[1,1,D]);
    KER = repmat(KER,[1,1,D]); 
    Den_KER = repmat(Den_KER,[1,1,D]);
end
Normin1 = conj(KER).*fft2(S); % 卷积核的共轭点乘图像S fft2将S转换到频域
%% 
beta = 2*lambda;
while beta < betamax
    Denormin   = Den_KER + beta*Denormin2;
    h = [diff(S,1,2), S(:,1,:) - S(:,end,:)]; % 计算水平方向的梯度
    v = [diff(S,1,1); S(1,:,:) - S(end,:,:)]; % 计算垂直方向的梯度
    if D==1
        t = (h.^2+v.^2)<lambda/beta;
    else
        t = sum((h.^2+v.^2),3)<lambda/beta; % 各通道模值相加到一起进行比较 返回true或者false
        t = repmat(t,[1,1,D]); % 再拓展回t原本的通道数
    end
    h(t)=0; v(t)=0; % 将变量h和v的掩膜t中对应位置为true的元素置为0
    Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)]; % 水平方向上的差分结果Normin2
    FS = (Normin1 + beta*fft2(Normin2))./Denormin;
    S = real(ifft2(FS)); % 傅里叶逆变换
    beta = beta*kappa;
end
S = S(1:H, 1:W, :);
end
