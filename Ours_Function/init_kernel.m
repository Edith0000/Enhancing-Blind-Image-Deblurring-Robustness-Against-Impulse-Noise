function [k] = init_kernel(minsize) %初始化卷积核
  k = zeros(minsize, minsize); %调用zeros函数生成一个minsize x minsize的全零矩阵
  k((minsize - 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
end