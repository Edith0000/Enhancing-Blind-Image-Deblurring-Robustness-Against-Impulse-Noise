function [px, py, threshold]= threshold_pxpy_v1(latent,psf_size,threshold) % 计算图像梯度阈值

% mask indicates region with gradients. outside of mask shud be smooth...

if ~exist('threshold', 'var')
    threshold = 0; % 若没有提供threshold，设为0
    b_estimate_threshold = true;
else
    b_estimate_threshold = false;
end
%{
denoised = bilateral_filter(latent, 2.0, param.sigma, 'replicate', 2);
if opts.verbose
    figure(1992);imshow(denoised);title('denoised');
end
%}
denoised=latent;

%%
% derivative filters 导数滤波器
dx = [-1 1; 0 0]; %水平
dy = [-1 0; 1 0]; %竖直
%%
% px = imfilter(denoised, [0 -1 1], 'same', 'replicate');
% py = imfilter(denoised, [0;-1;1], 'same', 'replicate');
px = conv2(denoised, dx, 'valid'); %拿dx跟latent卷求水平梯度
py = conv2(denoised, dy, 'valid'); %拿dy跟latent卷求竖直梯度
pm = px.^2 + py.^2; % 求梯度的幅值平方


% if this is the first prediction, then we need to find an appropriate
% threshold value by building a histogram of gradient magnitudes
if b_estimate_threshold %如果需要估计阈值
    pd = atan(py./px); % 求反正切值（角度）
    pm_steps = 0:0.00006:2;
    H1 = cumsum(flipud(histc(pm(pd >= 0 & pd < pi/4), pm_steps))); % cumsum函数（行累加）
    H2 = cumsum(flipud(histc(pm(pd >= pi/4 & pd < pi/2), pm_steps))); % histc函数（直方图统计）
    H3 = cumsum(flipud(histc(pm(pd >= -pi/4 & pd < 0), pm_steps))); % flipud 函数用于将直方图矩阵上下翻转，即将最小值放在最上面
    H4 = cumsum(flipud(histc(pm(pd >= -pi/2 & pd < -pi/4), pm_steps)));

    th = max([max(psf_size)*20, 10]); % 估计的阈值为模糊核尺寸的20倍或10中较大者
    
    for t=1:numel(pm_steps) %numel函数返回数组或者矩阵的元素个数
        min_h = min([H1(t) H2(t) H3(t) H4(t)]);
        if min_h >= th
            threshold = pm_steps(end-t+1);
            break
        end
    end % 估计阈值，讲最小的累积直方图值作为新阈值
end

% thresholding
m = pm < threshold;
while all(m(:)==1)
    threshold = threshold * 0.81;
    m = pm < threshold;
end
px(m) = 0;
py(m) = 0; %将满足阈值条件的梯度值清零


% % update prediction parameters
% threshold = threshold * 0.9;
%% my modification
if b_estimate_threshold
    threshold = threshold;
else
    threshold = threshold./1.1;
end % 根据逻辑真假调整阈值

end

