function [ k ] = kernel_solver_L4( Y,b,k_size,M,lambda,band)
% Solve kernel
% Written by Yuanchao Bai

dx=rot90([0,-1,1],2);
dy=dx';

% k_size_hf=floor(k_size/2);
% mask=zeros(size(Y));
% mask(k_size_hf+1:end-k_size_hf,k_size_hf+1:end-k_size_hf)=1;
% mask=ones(size(Y));

if isempty(M)
    M=ones(size(Y));
end


Yx=imfilter(Y,dx,'conv','replicate');
Yy=imfilter(Y,dy,'conv','replicate');

[index3]=AMF3(Yx);
data_we3=1-index3;
[index4]=AMF3(Yy);
data_we4=1-index4;

Yx=(data_we3.*Yx).*M;
Yy=(data_we4.*Yy).*M;



%Yx=(data_we.*Yx).*M;


bx=imfilter(b,dx,'conv','replicate');
by=imfilter(b,dy,'conv','replicate');
[index1]=AMF3(bx);
data_we1=1-index1;
[index2]=AMF3(by);
data_we2=1-index2;

bx=data_we1.*bx;
by=data_we2.*by;
pad_time=3;
pad_size=floor(k_size*pad_time);
bx_p=padarray(bx,[pad_size,pad_size],'both');
by_p=padarray(by,[pad_size,pad_size],'both');


Yx_p=padarray(Yx,[pad_size,pad_size],'both');
Yy_p=padarray(Yy,[pad_size,pad_size],'both');
%data_we=padarray(data_we,[pad_size,pad_size],0,'both');
% vx_p=padarray(vx,[pad_size,pad_size],'both');
% vy_p=padarray(vy,[pad_size,pad_size],'both'); 

Yx_f=fft2(Yx_p);
Yy_f=fft2(Yy_p);

bx_f=fft2(bx_p);
by_f=fft2(by_p);

% vx_f=fft2(vx_p);
% vy_f=fft2(vy_p);   

wx=25;
wy=25;
psf_size=[k_size,k_size];

b_f = wx*conj(Yx_f).*bx_f+wy*conj(Yy_f).*by_f;
b = real(otf2psf(b_f, psf_size));

p.m = wx*conj(Yx_f).*Yx_f+wy*conj(Yy_f).*Yy_f;
%p.img_size = size(blurred);
p.img_size = size(bx_f);
p.psf_size = psf_size;
p.lambda = lambda;
%p.data_we=data_we;
psf = ones(psf_size) / prod(psf_size);
psf = conjgrad(psf, b, 20, 1e-5, @compute_Ax, p);

psf(psf < max(psf(:))*0.05) = 0;
%     psf(psf < 0) = 0;    
k = psf / sum(psf(:));    
end

function y = compute_Ax(x, p)
    x_f = psf2otf(x, p.img_size);
    y = otf2psf((p.m .* x_f), p.psf_size);
    y = y + p.lambda * x;
end

function x=conjgrad(x,b,maxIt,tol,Ax_func,func_param,visfunc)
% conjgrad.m
%
%   Conjugate gradient optimization
%
%     written by Sunghyun Cho (sodomau@postech.ac.kr)
%
    r = b - Ax_func(x,func_param);
    p = r;
    rsold = sum(r(:).*r(:));

    for iter=1:maxIt
        Ap = Ax_func(p,func_param);
        alpha = rsold/sum(p(:).*Ap(:));
        x=x+alpha*p;
        if exist('visfunc', 'var')
            visfunc(x, iter, func_param);
        end
        r=r-alpha*Ap;
        rsnew=sum(r(:).*r(:));
        if sqrt(rsnew)<tol
            break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
end

