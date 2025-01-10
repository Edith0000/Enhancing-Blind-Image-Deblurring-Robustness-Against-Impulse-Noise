function [img_denoise] = median_filter(img_noise, N)
    [ROW, COL, ~] = size(img_noise);  % 获取图像的尺寸，包括通道数
    img_noise = im2double(img_noise);
    img_denoise = zeros(ROW, COL, 3);  % 创建与输入图像相同大小的空白画布

    for k = 1:3  % 遍历每个颜色通道
        for i = 1:ROW - (N-1)
            for j = 1:COL - (N-1)
                mask = img_noise(i:i+(N-1), j:j+(N-1), k);  % 提取当前颜色通道的滤波窗口
                s = sort(mask(:));
                img_denoise(i+(N-1)/2, j+(N-1)/2, k) = s((N*N+1)/2);  % 对当前颜色通道进行中值滤波
            end
        end
    end
end