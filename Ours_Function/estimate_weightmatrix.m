function [ ww ] = estimate_weightmatrix( blur, latent, psf, is_previous)
alpha=1.8e-3; beta=2e-4;
ww = ones(size(blur));
    if ~is_previous
        bb = fftconv(latent, psf);
        temp = (blur-bb).^2;
        w_matrix = 1./(1+exp((temp-alpha)/beta)); % 权重W的形式解
        %%% 处理异常值
        max_blur = max(blur(:));
        min_blur = min(blur(:));
        w_matrix(blur==min_blur) = 0; 
        w_matrix(blur==max_blur) = 0;
        %%% 
        w_matrix(w_matrix<=0)=0+eps; % 模糊图像 blur 中的最大值
        w_matrix(w_matrix>=1)=1-eps; % 模糊图像 blur 中的最小值
        ww = w_matrix;
       
        ww(bb>1) = 0;
        ww(bb<0) = 0;
        tw=ww.*log(ww)+(1-ww).*log(1-ww);
    end
end

