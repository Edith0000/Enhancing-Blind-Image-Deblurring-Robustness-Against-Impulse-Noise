function [index]=AMF3(img)
[m, n,~]=size(img);            %m,n为图像的行数和列数
% [~,~,~]=size(img);
img1=img;  
 num=0;
 band=0.000001;
% img2=img(:,:,2);  
% img3=img(:,:,3); 
%% 图像边缘扩展
%为保证边缘的像素点可以被采集到，必须对原图进行像素扩展。
%一般设置的最大滤波窗口为7，所以只需要向上下左右各扩展3个像素即可采集到边缘像素。
Nmax=1;        %确定最大向外扩展为3像素，即最大窗口为7*7
imgn=zeros(m+2*Nmax,n+2*Nmax);      %新建一个扩展后大小的全0矩阵
imgn(Nmax+1:m+Nmax,Nmax+1:n+Nmax)=img1;  %将原图覆盖在imgn的正中间
%下面开始向外扩展，即把边缘的像素向外复制
imgn(1:Nmax,Nmax+1:n+Nmax)=img1(1:Nmax,1:n);                 %扩展上边界
imgn(1:m+Nmax,n+Nmax+1:n+2*Nmax)=imgn(1:m+Nmax,n+1:n+Nmax);    %扩展右边界
imgn(m+Nmax+1:m+2*Nmax,Nmax+1:n+2*Nmax)=imgn(m+1:m+Nmax,Nmax+1:n+2*Nmax);    %扩展下边界
imgn(1:m+2*Nmax,1:Nmax)=imgn(1:m+2*Nmax,Nmax+1:2*Nmax);       %扩展左边界
% figure;imshow(uint8(imgn));
re=imgn;        %扩展之后的图像
index=zeros(size(re));
%% 得到不是噪声点的中值
for i=Nmax+1:m+Nmax
    for j=Nmax+1:n+Nmax
        r=1;                %初始向外扩张1像素，即滤波窗口大小为3
      
           
            
            W=[imgn(i-r:i+r,j-1);imgn(i-r:i+r,j+1);imgn(i-r:i-1,j);imgn(i+r:i+1,j)];
            
            temp_W=imgn(i,j);
             
            Imin=min(W);        %最小灰度值
            Imax=max(W);         %最大灰度值
            if temp_W<Imin-band|| temp_W>Imax+band 
              index(i,j)=1;
              num=num+1;
            end
            
            
        end
        
 %% 判断当前窗口内的中心像素是否为噪声，是就用前面得到的中值替换，否则不替换       
       
end
index=index(Nmax+1:m+Nmax,Nmax+1:n+Nmax);
noise_level=num/(n*m);
end