function [index]=AMF3(img)
[m, n,~]=size(img);            %m,nΪͼ�������������
% [~,~,~]=size(img);
img1=img;  
 num=0;
 band=0.000001;
% img2=img(:,:,2);  
% img3=img(:,:,3); 
%% ͼ���Ե��չ
%Ϊ��֤��Ե�����ص���Ա��ɼ����������ԭͼ����������չ��
%һ�����õ�����˲�����Ϊ7������ֻ��Ҫ���������Ҹ���չ3�����ؼ��ɲɼ�����Ե���ء�
Nmax=1;        %ȷ�����������չΪ3���أ�����󴰿�Ϊ7*7
imgn=zeros(m+2*Nmax,n+2*Nmax);      %�½�һ����չ���С��ȫ0����
imgn(Nmax+1:m+Nmax,Nmax+1:n+Nmax)=img1;  %��ԭͼ������imgn�����м�
%���濪ʼ������չ�����ѱ�Ե���������⸴��
imgn(1:Nmax,Nmax+1:n+Nmax)=img1(1:Nmax,1:n);                 %��չ�ϱ߽�
imgn(1:m+Nmax,n+Nmax+1:n+2*Nmax)=imgn(1:m+Nmax,n+1:n+Nmax);    %��չ�ұ߽�
imgn(m+Nmax+1:m+2*Nmax,Nmax+1:n+2*Nmax)=imgn(m+1:m+Nmax,Nmax+1:n+2*Nmax);    %��չ�±߽�
imgn(1:m+2*Nmax,1:Nmax)=imgn(1:m+2*Nmax,Nmax+1:2*Nmax);       %��չ��߽�
% figure;imshow(uint8(imgn));
re=imgn;        %��չ֮���ͼ��
index=zeros(size(re));
%% �õ��������������ֵ
for i=Nmax+1:m+Nmax
    for j=Nmax+1:n+Nmax
        r=1;                %��ʼ��������1���أ����˲����ڴ�СΪ3
      
           
            
            W=[imgn(i-r:i+r,j-1);imgn(i-r:i+r,j+1);imgn(i-r:i-1,j);imgn(i+r:i+1,j)];
            
            temp_W=imgn(i,j);
             
            Imin=min(W);        %��С�Ҷ�ֵ
            Imax=max(W);         %���Ҷ�ֵ
            if temp_W<Imin-band|| temp_W>Imax+band 
              index(i,j)=1;
              num=num+1;
            end
            
            
        end
        
 %% �жϵ�ǰ�����ڵ����������Ƿ�Ϊ�������Ǿ���ǰ��õ�����ֵ�滻�������滻       
       
end
index=index(Nmax+1:m+Nmax,Nmax+1:n+Nmax);
noise_level=num/(n*m);
end