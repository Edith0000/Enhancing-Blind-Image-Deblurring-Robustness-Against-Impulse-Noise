function k=resizeKer(k,ret,k1,k2) % 调整模糊核的尺寸大小
% levin's code
k=imresize(k,ret); % 调用imresize函数对初始的模糊核尺寸进行放缩
k=max(k,0); % 将负值全替换为零
k=fixsize(k,k1,k2);
if max(k(:))>0
    k=k/sum(k(:));
end
end