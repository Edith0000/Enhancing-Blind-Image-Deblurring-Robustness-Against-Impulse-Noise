function [ k_estimate, Y_r_rgtv_cg ] =deblur_denoise_main4( Y_b,k_estimate_size, show_intermediate,opts)
% Blind image deblurring from coarse to fine

% Y_b: a blurred image
% k_estimate_size: the estimated size of kernel

% k_estimate: restored kernel
% Y_r_rgtv_cg: restored skeleton image

% Written by Yuanchao Bai
%noise_threshold=opts.noise_threshold;  

% scale_factor=log2(3);
% level_num=ceil(log(k_estimate_size/7)/log(scale_factor))+1;
% image_pyramid=cell(level_num,1);
% k_size=zeros(level_num,1);
% image_size=zeros(level_num,2);

%threshold=opts.threshold;
mu=opts.mu;
lambda=opts.lambda;
sigma=opts.sigma;
rho_mu=opts.rho_mu;
rho_lambda=opts.lambda;

ret = sqrt(0.5);


%%最大迭代级别
maxitr=max(floor(log(5/min(k_estimate_size))/log(ret)),0); %%floor函数（向下取整）；如果迭代次数小于0则将其设置为零
level_num = maxitr + 1;
%fprintf('Maximum iteration level is %d\n', level_num);  %%打印输出最大迭代次数
%%
retv=ret.^(0:maxitr); %不同尺度下的放缩因子
k1list=ceil(k_estimate_size*retv); %ceil函数（向上取整）
k1list=k1list+(mod(k1list,2)==0); % 修正k1list，若为偶数则加一,后续初始化卷积核有用
k2list=ceil(k_estimate_size*retv);
k2list=k2list+(mod(k2list,2)==0); % 生成一个和k1一样的列表

% if  threshold<0.001
%   s=2;
%   band=threshold;    
% elseif threshold<0.01
 s=2;
%   band=0.25*threshold; 
%   
%  elseif  threshold<0.05
%   s=2;
%   band=0.05*threshold;
%   
% elseif threshold<0.1
%   s=2;
%   band=0.01*threshold;
%    
% 
% elseif threshold<0.2
%    s=3;
%    band=0.001*threshold;
% 
% end
%image_pyramid{1}=TV_denoising(Y_b,[0.01,0.01],10);
% image_pyramid{1}=Y_b;
% k_size(1)=k_estimate_size;
% image_size(1,:)=size(image_pyramid{1});
% for i=2:level_num
%     image_size(i,:)=floor(image_size(i-1,:)/log2(3));
%     image_pyramid{i}=imresize(image_pyramid{i-1},image_size(i,:),'bilinear');
%     k_size(i)=floor(k_size(i-1)/log2(3));
%     k_size(i)=k_size(i)+(1-mod(k_size(i),2));
% end



frame=1;
Level=1;
[D,R]=GenerateFrameletFilter(frame);
KD  = @(x) FraDecMultiLevel2D(x,D,Level); % kernel decomposition
KF = @(x,ratio) kernel_filter(x,R,Level,ratio); % kernel filter

%% RGTV Blind



for level=level_num:-1:s
     
    mu=opts.mu;
    lambda=opts.lambda;
     cret=retv(level);
     image_pyramid{level}=downSmpImC(Y_b,cret); %对模糊图像进行下采样
     image_pyramid{level} = imfilter(image_pyramid{level},fspecial('gaussian',5,1),'same','replicate');
      
    

    if level == level_num
      % at coarsest level, initialize kernel（在一个最粗的层级上初始化卷积核）
      k_estimate = init_kernel(k1list(level)); %调用init_kernel函数初始化一个卷积核
      k1 = k1list(level);
      k2 = k1; % always square kernel assume
   else
    % upsample kernel from previous level to next finer level
    % （从上一级别 上采样卷积核 到 下一个更精细的级别）
    k1 = k1list(level);
    k2 = k1; % always square kernel assumed
    % resize kernel from previous level
    % （从上一级别调整卷积核的大小）
    k_estimate= resizeKer(k_estimate,1/ret,k1list(level),k2list(level)); %调用resizeKer函数
    end
        
    k_size(level)=size(k_estimate,1);
    [Y_b_padding,padsize] = G_padding(image_pyramid{level},k_estimate,1);
    Y_r_rgtv_cg=Y_b_padding;
    [h,w]=size(Y_r_rgtv_cg);
        

        
     

        for iter=1:3
           
            mu=opts.mu;
           lambda=opts.lambda;
            %fprintf('level %d, Iter %d:',level,iter);
            W1=ones(h*w,4);
            W=W1;
            for i=1:3
             %   fprintf('%d...',i);
                for j=1:3
                    Y_r_rgtv_cg=Deblur_GL_CG_4(Y_b_padding,k_estimate,W,mu,20);
                    W=W1.*weights_computation( Y_r_rgtv_cg,[],4,2 );
                end
                W1=weights_computation( Y_r_rgtv_cg,sigma,4,1 );
                W=W1.*weights_computation( Y_r_rgtv_cg,[],4,2 );
            end
           % fprintf('\n');

            Y_r_rgtv_cg=Y_r_rgtv_cg(padsize(1)+1:end-padsize(1),padsize(2)+1:end-padsize(2));
             
%                [~,rat_1]=AMF2(Y_r_rgtv_cg,band);
%    if rat_1>0.01
%        Y_r_rgtv_cg=medfilt2(Y_r_rgtv_cg); 
%    end
            if show_intermediate
                figure(11);
                imshow(Y_r_rgtv_cg);
                title('RGTV CG');
            end

            %% kernel estimate
            t_s=0.1;
            t_r=0.3;
            [ M ] = informative_edge_mask_adaptive_mine( Y_r_rgtv_cg,t_s,t_r,5);
            k_estimate=kernel_solver_L2(Y_r_rgtv_cg,image_pyramid{level},k_size(level),M,lambda);

            if level<=2
            Cf=KD(k_estimate);
            k_estimate=KF(Cf,0.05); % filter noise in restored kernel
            k_estimate(k_estimate<max(k_estimate(:))*0.05)=0; % Thresholding
            k_estimate=k_estimate/sum(k_estimate(:));
            [ k_estimate ] = kernel_centralize(k_estimate,0.1);
            end

            if show_intermediate
                figure(12);
                imshow(k_estimate,[]);
                title('Estimated kernel');
            end

            mu=mu/ rho_mu;   %  1.1;
            lambda=lambda/rho_lambda; % 1.2;
        end
        
          
    
        
        
        
end
  % org_noise_threshold=noise_threshold;
 for level=s-1:-1:1
  mu=opts.mu;
lambda=opts.lambda;
   

    cret=retv(level);
    image_pyramid{level}=downSmpImC(Y_b,cret); %对模糊图像进行下采样
    
     if level == level_num
      % at coarsest level, initialize kernel（在一个最粗的层级上初始化卷积核）
      k_estimate = init_kernel(k1list(level)); %调用init_kernel函数初始化一个卷积核
      k1 = k1list(level);
      k2 = k1; % always square kernel assume
  else
    % upsample kernel from previous level to next finer level
    % （从上一级别 上采样卷积核 到 下一个更精细的级别）
    k1 = k1list(level);
    k2 = k1; % always square kernel assumed
    
    % resize kernel from previous level
    % （从上一级别调整卷积核的大小）
    k_estimate= resizeKer(k_estimate,1/ret,k1list(level),k2list(level)); %调用resizeKer函数
     end
 
    k_size(level)=size(k_estimate,1);
    [index]=AMF3(image_pyramid{level});
    
    [h1,w1]=size(image_pyramid{level});
    [Y_b_padding,padsize] = G_padding(image_pyramid{level},k_estimate,1);
    kk1=size(k_estimate,1);
    Y_r_rgtv_cg=Y_b_padding;
    [h,w]=size(Y_r_rgtv_cg);
    ww=ones(h,w);
    ww(kk1+1:kk1+h1,kk1+1:kk1+w1)=1-index;
    
    
    figure(20)
    imshow(ww,[])
    for iter=1:3

      %  fprintf('level %d, Iter %d:',level,iter);
        W1=ones(h*w,4);
        W=W1;
        for i=1:3
            %fprintf('%d...',i);
            for j=1:3
                Y_r_rgtv_cg=Deblur_GL_CG_5(Y_b_padding,k_estimate,W,mu,20,ww);
                W=W1.*weights_computation( Y_r_rgtv_cg,[],4,2 );
            end
            W1=weights_computation( Y_r_rgtv_cg,sigma,4,1 );
            W=W1.*weights_computation( Y_r_rgtv_cg,[],4,2 );
        end
      %  fprintf('\n');

  Y_r_rgtv_cg=Y_r_rgtv_cg(padsize(1)+1:end-padsize(1),padsize(2)+1:end-padsize(2));
%    [~,rat_1]=AMF2(Y_r_rgtv_cg,band);
%    if rat_1>0.01
%        Y_r_rgtv_cg=medfilt2(Y_r_rgtv_cg); 
%    end
        %Y_r_rgtv_cg=medfilt2( Y_r_rgtv_cg);
       
        
        if show_intermediate
            figure(11);
            imshow(Y_r_rgtv_cg);
            title('RGTV CG');
        end

        %% kernel estimate
        t_s=0.1;
        t_r=0.3;
        [ M ] = informative_edge_mask_adaptive_mine(Y_r_rgtv_cg,t_s,t_r,5);
        k_estimate=kernel_solver_L5(Y_r_rgtv_cg,image_pyramid{level},k_size(level),M,lambda);

        if level<=2
            Cf=KD(k_estimate);
            k_estimate=KF(Cf,0.05); % filter noise in restored kernel
            k_estimate(k_estimate<max(k_estimate(:))*0.05)=0; % Thresholding
            k_estimate=k_estimate/sum(k_estimate(:));
            [ k_estimate ] = kernel_centralize(k_estimate,0.1);
            
        else
           k_estimate( k_estimate(:) < max( k_estimate(:))/20) = 0; 
            k_estimate=  k_estimate / sum( k_estimate(:));
            
        end

        if show_intermediate
            figure(12);
            imshow(k_estimate,[]);
            title('Estimated kernel');
        end

     
            mu=mu/ rho_mu;   %  1.1;
            lambda=lambda/rho_lambda; % 1.2;
    end
end
    
        
       
       
 
end






