function noisyImage = imnoise_impuls(image, noiseDensity)
    % image: 输入的图像
    % noiseDensity: 噪声密度，即椒盐噪声的比例
    
    % 定义噪声强度范围
    intensityRange = [0, 1];
    
    % 复制输入图像到输出图像
    noisyImage = image;
    
    % 获取图像的尺寸
    [m, n] = size(noisyImage);
    numPixels = m * n;
    
    % 计算需要添加的椒盐噪声像素数量
    numNoisyPixels = round(numPixels * noiseDensity);
    
    % 生成随机索引和强度值
    randomIndices = randperm(numPixels, numNoisyPixels);
    randomIntensities = randi(intensityRange, [numNoisyPixels, 1]);
    
    % 在随机位置添加椒盐噪声
    noisyImage(randomIndices) = randomIntensities;
end
