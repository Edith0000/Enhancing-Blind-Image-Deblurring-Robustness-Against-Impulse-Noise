function [kernel, interim_latent] = blind_deconv(y, lambda_grad, opts)
% gamma correct����У������ͼ��������Աȶ�)
if opts.gamma_correct~=1
    y = y.^opts.gamma_correct;
end
%%��������
ret = sqrt(0.5);
%%����������
maxitr=max(floor(log(5/min(opts.kernel_size))/log(ret)),0); %%floor����������ȡ�����������������С��0��������Ϊ��
num_scales = maxitr + 1;
fprintf('Maximum iteration level is %d\n', num_scales);  %%��ӡ�������������
%%
retv=ret.^(0:maxitr); %��ͬ�߶��µķ�������
k1list=ceil(opts.kernel_size*retv); %ceil����������ȡ����
k1list=k1list+(mod(k1list,2)==0); % ����k1list����Ϊż�����һ,������ʼ�����������
k2list=ceil(opts.kernel_size*retv);
k2list=k2list+(mod(k2list,2)==0); % ����һ����k1һ�����б�

% blind deconvolution - multiscale processing��ä����-��߶ȴ���
for s = num_scales:-1:1
  if (s == num_scales)
      % at coarsest level, initialize kernel����һ����ֵĲ㼶�ϳ�ʼ������ˣ�
      ks = init_kernel(k1list(s)); %����init_kernel������ʼ��һ�������
      k1 = k1list(s);
      k2 = k1; % always square kernel assume
  else
    % upsample kernel from previous level to next finer level
    % ������һ���� �ϲ�������� �� ��һ������ϸ�ļ���
    k1 = k1list(s);
    k2 = k1; % always square kernel assumed
    
    % resize kernel from previous level
    % ������һ�����������˵Ĵ�С��
    ks = resizeKer(ks,1/ret,k1list(s),k2list(s)); %����resizeKer����
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  cret=retv(s);
  ys=downSmpImC(y,cret); %��ģ��ͼ������²���

  fprintf('Processing scale %d/%d; kernel size %dx%d; image size %dx%d\n', ...
            s, num_scales, k1, k2, size(ys,1), size(ys,2));
  %-----------------------------------------------------------%
  %% Useless operation
  if (s == num_scales)
    [~, ~, threshold]= threshold_pxpy_v1(ys,max(size(ks))); % ����threshold_pxpy_v1���������ݶ���ֵthreshold
  end
  %-----------------------------------------------------------%
  if s <= 1
      opts.xk_iter = opts.last_iter; % ������������Ϊ����������
      [ks, lambda_grad, interim_latent] = fine_deblur(ys, ks,lambda_grad, threshold, opts); % ���о�ϸ��ȥģ������
  else
    if opts.isnoisy
        ys = imfilter(ys,fspecial('gaussian',5,1),'same','replicate'); % ���opts.isnoisyΪ�棬��ʹ�ø�˹�˲���������ͼ��ys����ƽ������
    end
    [ks, lambda_grad, interim_latent] = coarse_deblur(ys, ks,lambda_grad, threshold, opts); % ���д��Ե�ȥģ������
  end
   %% center the kernel����������˵�����λ�ã������й�һ������
   ks = adjust_psf_center(ks);
   ks(ks(:)<0) = 0; % ������С���������ֵ����Ϊ��
   sumk = sum(ks(:)); % ������������ֵ���ܺ�
   ks = ks./sumk; % ��һ������ʹ����������ֵ֮��Ϊ1
  %% set elements below threshold to 0����С����ֵ��Ԫ������Ϊ�㣬������ϡ���Ի�ȥ��һЩ������
  if (s == 1)
    kernel = ks;
    if opts.k_thresh>0
        kernel(kernel(:) < max(kernel(:))/opts.k_thresh) = 0; % ��С����ֵ��Ԫ������Ϊ��
    else
        kernel(kernel(:) < 0) = 0; % ���������С�����Ԫ������Ϊ��
    end
    kernel = kernel / sum(kernel(:)); %������˽��й�һ����ʹ������Ԫ��֮��Ϊ1
  end
end
end
%% Sub-function���Ӻ�����
function [k] = init_kernel(minsize) %��ʼ�������
  k = zeros(minsize, minsize); %����zeros��������һ��minsize x minsize��ȫ�����
  k((minsize - 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
end

%%
function sI=downSmpImC(I,ret) %��ͼ������²�����ret���������ӣ�
%% refer to Levin's code
if (ret==1) 
    sI=I;
    return % ֱ��ԭͼ���
end
%%%%%%%%%%%%%%%%%%%

sig=1/pi*ret; % ���ݱ�������ret�����˹�˲����ı�׼��sig

g0=(-50:50)*2*pi;
sf=exp(-0.5*g0.^2*sig^2); % ����һ����˹�˲������˴�СΪ[-50:50]
sf=sf/sum(sf); % ��һ��
csf=cumsum(sf); % cumsum������������е��ۼӺͣ�
csf=min(csf,csf(end:-1:1));
ii=csf>0.05;

sf=sf(ii); % ��������ֵii��sf����ȡ��Ӧ��Ԫ�أ��õ��µĸ�˹�˲�����sf
sum(sf); 

I=conv2(sf,sf',I,'valid'); % ��ͼ��I���о������

[gx,gy]=meshgrid(1:1/ret:size(I,2),1:1/ret:size(I,1)); %������������gx��gy

sI=interp2(I,gx,gy,'bilinear'); % ����˫���Բ�ֵ�ķ�������ͼ����в�ֵ����
end
%%
function k=resizeKer(k,ret,k1,k2) % ����ģ���˵ĳߴ��С
% levin's code
k=imresize(k,ret); % ����imresize�����Գ�ʼ��ģ���˳ߴ���з���
k=max(k,0); % ����ֵȫ�滻Ϊ��
k=fixsize(k,k1,k2);
if max(k(:))>0
    k=k/sum(k(:));
end
end
%% 
function nf=fixsize(f,nk1,nk2) % ͨ��ɾ���򲹳������к��У����������Ϊָ����Ŀ��ߴ�
[k1,k2]=size(f);

while((k1~=nk1)||(k2~=nk2))
    
    if (k1>nk1)
        s=sum(f,2); %ÿһ�н������{sum��a��dim��dim=1������ͣ���2������ͣ�}
        if (s(1)<s(end))
            f=f(2:end,:); %��һ��Ԫ��֮��С�����һ�У�ɾ����һ��
        else
            f=f(1:end-1,:); %��һ��Ԫ��֮�ʹ������һ�У�ɾ�����һ��
        end
    end
    
    if (k1<nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            tf=zeros(k1+1,size(f,2));
            tf(1:k1,:)=f; % ��һ��Ԫ��֮��С�����һ�У��ھ���Ķ������һ�У������Ϊ��
            f=tf;
        else
            tf=zeros(k1+1,size(f,2));
            tf(2:k1+1,:)=f; % ��һ��Ԫ��֮�ʹ������һ�У��ھ���ĵײ����һ�У������Ϊ��
            f=tf;
        end
    end
    %���н�������һ���Ĳ���
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
    
[k1,k2]=size(f); % ģ���˵ĳߴ�ĳ��Ϳ�

end

nf=f;
end