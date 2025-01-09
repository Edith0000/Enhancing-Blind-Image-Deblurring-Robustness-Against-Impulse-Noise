clc;
clear;
close all;

addpath(genpath('./Dong_Function'));

opts.Ik_iter = 4;
opts.display = 1;
%opts.kernel_size = 85;
opts.lambda_grad = 0.008; 
opts.k_reg_wt = 0.1;
opts.gamma_correct = 1.0;
opts.sigma = 5/255;

% 设置文件夹路径
blur_noise_folder = './Data/Mixed_data/blur_noise_image';
kernel_folder = './Results/Mixed_results/Dong_et_all/kernel';
result_folder = './Results/Mixed_results/Dong_et_all/result';

% 获取blur_noise_image文件夹中的子文件夹列表
subfolders = dir(blur_noise_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
opts.gamma_correct = 1.0;
% 保存去模糊时间的结构体数组
processing_data = struct('image_name', {}, 'processing_time', {});

% 遍历每个子文件夹i=iiiii
for i = 7:length(subfolders)
    subfolder_name = subfolders(i).name;
    subfolder_path = fullfile(blur_noise_folder, subfolder_name);

    % 创建kernel文件夹中对应子文件夹
    kernel_subfolder_path = fullfile(kernel_folder, subfolder_name);
    if ~isfolder(kernel_subfolder_path)
        mkdir(kernel_subfolder_path);
    end

    % 创建result文件夹中对应子文件夹
    result_subfolder_path = fullfile(result_folder, subfolder_name);
    if ~isfolder(result_subfolder_path)
        mkdir(result_subfolder_path);
    end
    % 获取子文件夹中的模糊图像的图像列表
    B_N_files = dir(fullfile(subfolder_path, '*.png'));
    B_N_files = {B_N_files.name};

    % 遍历子文件夹中的每张图像
    for j = 1:length(B_N_files)
        image_name = B_N_files{j};
        image_path = fullfile(subfolder_path, image_name);

        kernel_label=ceil(mod(j-1, 8) + 1/8);
        kernel_size=[19 17 15 27 13 21 23 23];
        opts.kernel_size=kernel_size(kernel_label);

        I=imread(image_path);  % 模糊带噪声图像
        if numel(size(I))>2
            Y_b=im2double(rgb2gray(I));
        else
            Y_b=im2double(I);
        end

        y = Y_b;
        tic
        [kernel, latent,weight_k,weight_x] = blind_deconv_main_ours(y, opts);
        processing_time=toc;

        % 保存模糊核估计时间
        processing_data(end+1).image_name = image_name;
        processing_data(end).processing_time = processing_time;

%         k = kernel - min(kernel(:));
%         k_out = k./max(k(:));
        kernel_name = strcat(image_name, '_kernel.png');
        kernel_path = fullfile(kernel_subfolder_path, kernel_name);
        imwrite(k_out, kernel_path);
        deblurImg=im2double(latent);
        result_name = strcat(image_name, '_result.png');
        result_path = fullfile(result_subfolder_path, result_name);
        imwrite(deblurImg, result_path);
        disp([image_name ' 已完成处理.']);
    end
end
save('./Results/Mixed_results/Dong_et_all/Processing_data.mat', 'processing_data');
rmpath('./Dong_Function')