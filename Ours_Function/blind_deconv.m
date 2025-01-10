function [kernel, interim_latent] = blind_deconv(y, lambda_grad, opts)
% gamma correct（γ校正调整图像亮度与对比度)
if opts.gamma_correct~=1
    y = y.^opts.gamma_correct;
end
%%比例因子
ret = sqrt(0.5);
%%最大迭代级别
maxitr=max(floor(log(5/min(opts.kernel_size))/log(ret)),0); %%floor函数（向下取整）；如果迭代次数小于0则将其设置为零
num_scales = maxitr + 1;
fprintf('Maximum iteration level is %d\n', num_scales);  %%打印输出最大迭代次数
%%
retv=ret.^(0:maxitr); %不同尺度下的放缩因子
k1list=ceil(opts.kernel_size*retv); %ceil函数（向上取整）
k1list=k1list+(mod(k1list,2)==0); % 修正k1list，若为偶数则加一,后续初始化卷积核有用
k2list=ceil(opts.kernel_size*retv);
k2list=k2list+(mod(k2list,2)==0); % 生成一个和k1一样的列表

% blind deconvolution - multiscale processing（盲解卷积-多尺度处理）
for s = num_scales:-1:1
  if (s == num_scales)
      % at coarsest level, initialize kernel（在一个最粗的层级上初始化卷积核）
      ks = init_kernel(k1list(s)); %调用init_kernel函数初始化一个卷积核
      k1 = k1list(s);
      k2 = k1; % always square kernel assume
  else
    % upsample kernel from previous level to next finer level
    % （从上一级别 上采样卷积核 到 下一个更精细的级别）
    k1 = k1list(s);
    k2 = k1; % always square kernel assumed
    
    % resize kernel from previous level
    % （从上一级别调整卷积核的大小）
    ks = resizeKer(ks,1/ret,k1list(s),k2list(s)); %调用resizeKer函数
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  cret=retv(s);
  ys=downSmpImC(y,cret); %对模糊图像进行下采样

  fprintf('Processing scale %d/%d; kernel size %dx%d; image size %dx%d\n', ...
            s, num_scales, k1, k2, size(ys,1), size(ys,2));
  %-----------------------------------------------------------%
  %% Useless operation
  if (s == num_scales)
    [~, ~, threshold]= threshold_pxpy_v1(ys,max(size(ks))); % 调用threshold_pxpy_v1函数计算梯度阈值threshold
  end
  %-----------------------------------------------------------%
  if s <= 1
      opts.xk_iter = opts.last_iter; % 将迭代次数设为最大迭代次数
      [ks, lambda_grad, interim_latent] = fine_deblur(ys, ks,lambda_grad, threshold, opts); % 进行精细的去模糊处理
  else
    if opts.isnoisy
        ys = imfilter(ys,fspecial('gaussian',5,1),'same','replicate'); % 如果opts.isnoisy为真，则使用高斯滤波器对输入图像ys进行平滑处理
    end
    [ks, lambda_grad, interim_latent] = coarse_deblur(ys, ks,lambda_grad, threshold, opts); % 进行粗略的去模糊处理
  end
   %% center the kernel（调整卷积核的中心位置，并进行归一化处理）
   ks = adjust_psf_center(ks);
   ks(ks(:)<0) = 0; % 将中心小于零的像素值设置为零
   sumk = sum(ks(:)); % 计算所有像素值的总和
   ks = ks./sumk; % 归一化处理，使得所有像素值之和为1
  %% set elements below threshold to 0（将小于阈值的元素设置为零，控制其稀疏性或去除一些噪声）
  if (s == 1)
    kernel = ks;
    if opts.k_thresh>0
        kernel(kernel(:) < max(kernel(:))/opts.k_thresh) = 0; % 将小于阈值的元素设置为零
    else
        kernel(kernel(:) < 0) = 0; % 将卷积核中小于零的元素设置为零
    end
    kernel = kernel / sum(kernel(:)); %将卷积核进行归一化，使其所有元素之和为1
  end
end
end
%% Sub-function（子函数）
function [k] = init_kernel(minsize) %初始化卷积核
  k = zeros(minsize, minsize); %调用zeros函数生成一个minsize x minsize的全零矩阵
  k((minsize - 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
end

%%
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
%%
function k=resizeKer(k,ret,k1,k2) % 调整模糊核的尺寸大小
% levin's code
k=imresize(k,ret); % 调用imresize函数对初始的模糊核尺寸进行放缩
k=max(k,0); % 将负值全替换为零
k=fixsize(k,k1,k2);
if max(k(:))>0
    k=k/sum(k(:));
end
end
%% 
function nf=fixsize(f,nk1,nk2) % 通过删除或补充矩阵的行和列，将矩阵调整为指定的目标尺寸
[k1,k2]=size(f);

while((k1~=nk1)||(k2~=nk2))
    
    if (k1>nk1)
        s=sum(f,2); %每一行进行求和{sum（a，dim）dim=1（列求和）或2（行求和）}
        if (s(1)<s(end))
            f=f(2:end,:); %第一行元素之和小于最后一行，删除第一行
        else
            f=f(1:end-1,:); %第一行元素之和大于最后一行，删除最后一行
        end
    end
    
    if (k1<nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            tf=zeros(k1+1,size(f,2));
            tf(1:k1,:)=f; % 第一行元素之和小于最后一行，在矩阵的顶部添加一行，并填充为零
            f=tf;
        else
            tf=zeros(k1+1,size(f,2));
            tf(2:k1+1,:)=f; % 第一行元素之和大于最后一行，在矩阵的底部添加一行，并填充为零
            f=tf;
        end
    end
    %对列进行与行一样的操作
    if (k2>nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            f=f(:,2:end);
        else
            f=f(:,1:end-1);
        end
    end
    
    if (k2<nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            tf=zeros(size(f,1),k2+1);
            tf(:,1:k2)=f;
            f=tf;
        else
            tf=zeros(size(f,1),k2+1);
            tf(:,2:k2+1)=f;
            f=tf;
        end
    end
    
[k1,k2]=size(f); % 模糊核的尺寸的长和宽

end

nf=f;
end