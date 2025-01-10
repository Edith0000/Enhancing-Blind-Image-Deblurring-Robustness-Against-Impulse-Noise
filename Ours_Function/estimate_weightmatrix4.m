function [ ww ] = estimate_weightmatrix4(blurred_w, latent_w, psf, is_previous,opts,pre_W,index)
% opts.alpha1=1.8e-3; opts.beta1=0.5*abs(opts.alpha1/log(0.5));
% %opts.alpha=1.5;
% %opts.alpha=0.1;


    if ~is_previous
        bb = fftconv(latent_w, psf);

        temp = (blurred_w-bb).^2;

        w_matrix = 1./(1+exp((temp-opts.alpha1*(index-pre_W))./opts.beta1)); % 权重W的形式解

           
        max_blur = max(blurred_w(:));
        min_blur = min(blurred_w(:));
        w_matrix(blurred_w==min_blur) = 0; 
        w_matrix(blurred_w==max_blur) = 0;
        %%% 
        w_matrix(w_matrix<=0)=0+eps; % 模糊图像 blur 中的最大值
        w_matrix(w_matrix>=1)=1-eps; % 模糊图像 blur 中的最小值
        ww = w_matrix;
       
        ww(bb>1) = 0;
        ww(bb<0) = 0;
       % tw=ww.*log(ww)+(1-ww).*log(1-ww);
    end
end

