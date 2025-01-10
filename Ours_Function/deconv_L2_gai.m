function latent = deconv_L2(blurred, latent0, psf, data_we, L2_we, ...
                                          weight_x, weight_y, weight_xx, weight_yy, weight_xy)
%L2正则化约束
% deconv_L2.m
%
% Deconvolution using a Gaussian prior
%
% written by Sunghyun Cho (sodomau@postech.ac.kr)
%                          
%% 修改
imageDims = ndims(blurred); % 获取维度
if imageDims == 2
    Bn = medfilt2(blurred, [3 3]); % 对二维图像进行中值滤波 灰度图
else
    Channels = size(blurred, 3); % 获取图像的通道数
    % 对每个通道分别进行中值滤波
    filteredChannels = zeros(size(blurred));
    for channel = 1:Channels
        filteredChannels(:,:,channel) = medfilt2(blurred(:,:,channel), [3 3]);
    end
    Bn = im2double(filteredChannels); %彩色
end
%% 完毕
    %size(blurred)
    %Bn = medfilt2(blurred);

    if ~exist('weight_x', 'var')
        weight_x = ones(size(blurred), 'single');
        weight_y = ones(size(blurred), 'single');
        weight_xx = zeros(size(blurred), 'single');
        weight_yy = zeros(size(blurred), 'single');
        weight_xy = zeros(size(blurred), 'single');
    end
    
    img_size = size(blurred);

    dxf=[0 -1 1];
    dyf=[0 -1 1]';
    dyyf=[-1; 2; -1];
    dxxf=[-1, 2, -1];
    dxyf=[-1 1 0;1 -1 0; 0 0 0];
    
    latent = latent0;
    psf_f = single(psf2otf(psf, img_size)); % 将模糊核转换到频域
    % compute b
    b = real(ifft2(fft2(data_we.*blurred) .* conj(psf_f))); % conj（）求共轭
    %% 添加
    mu = 6e-7; % 添加
    b = b+mu*Bn; % 添加
    % 完毕
    b = b(:);

    % set x
    x = latent(:);

    % run conjugate gradient(共轭梯度法）
    cg_param.psf = psf;
    cg_param.L2_we = L2_we;
    cg_param.data_we = data_we;
    cg_param.img_size = img_size;
    cg_param.psf_f = psf_f; % kernel的频域
    cg_param.weight_x = weight_x;
    cg_param.weight_y = weight_y;
    cg_param.weight_xx = weight_xx;
    cg_param.weight_yy = weight_yy;
    cg_param.weight_xy = weight_xy;
    cg_param.dxf = dxf;
    cg_param.dyf = dyf;
    cg_param.dxxf = dxxf;
    cg_param.dyyf = dyyf;
    cg_param.dxyf = dxyf;
    cg_param.latent = latent; % 添加
    cg_param.Bn = Bn;% 添加
    cg_param.mu = mu;

    x = conjgrad(x, b, 25, 1e-4, @Ax, cg_param); %, @vis); % 共轭梯度法求解x
    
    latent = reshape(x, img_size);
end

function y = Ax(x, p) % x y 就是要求的I
    x = reshape(x, p.img_size);
    x_f = fft2(x); % 进行傅里叶变换
    y = real(ifft2(fft2(p.data_we.*real(ifft2(p.psf_f.*x_f))).*conj(p.psf_f))); % conj()求共轭
    %% 添加
    y = y+p.mu*ones(size(y));
    %% 完毕
 % 梯度的正则化项
    y = y + p.L2_we*imfilter(p.weight_x.*imfilter(x, p.dxf, 'circular'), p.dxf, 'conv', 'circular');
    y = y + p.L2_we*imfilter(p.weight_y.*imfilter(x, p.dyf, 'circular'), p.dyf, 'conv', 'circular');
    y = y + p.L2_we*imfilter(p.weight_xx.*imfilter(x, p.dxxf, 'circular'), p.dxxf, 'conv', 'circular');
    y = y + p.L2_we*imfilter(p.weight_yy.*imfilter(x, p.dyyf, 'circular'), p.dyyf, 'conv', 'circular');
    y = y + p.L2_we*imfilter(p.weight_xy.*imfilter(x, p.dxyf, 'circular'), p.dxyf, 'conv', 'circular');
 %% LX添加的正则项
    %bn = single(x)./255;
    %Blur = min(max(bn, -10), 20); % preprocessing
    %y_prime = medfilt2(min(max(Blur, 0), 1));
    %K_I = imfilter(p.latent, p.psf, 'circular', 'conv');
    %mu = 9e-3; % 要调的
    %reg_term = mu*(K_I - y_prime).^2;
    %y = y + reg_term;  % 添加正则化项
 %% 添加完毕
    y = y(:);
end

