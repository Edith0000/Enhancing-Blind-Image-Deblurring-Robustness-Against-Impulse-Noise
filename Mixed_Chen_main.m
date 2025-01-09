clc;
clear;
close all; 

addpath(genpath('./Chen_Function/'));

opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;
%opts.kernel_size = 27;
saturation = 0;
lambda_lmg =4e-3; lambda_grad =4e-3;opts.gamma_correct = 1;
lambda_tv = 0.001; lambda_l0 = 5e-4; weight_ring = 1;

% 设置文件夹路径
blur_noise_folder = './Data/Mixed_data/blur_noise_image';
kernel_folder = './Results/Mixed_results/Chen_et_all/kernel';
result_folder = './Results/Mixed_results/Chen_et_all/result';

% 获取blur_noise_image文件夹中的子文件夹列表
subfolders = dir(blur_noise_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

% 保存去模糊时间的结构体数组
processing_data = struct('image_name', {}, 'processing_time', {});

% 遍历每个子文件夹i=iiiii
for i = 1:length(subfolders)
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

        yg = Y_b;

        tic;
        [kernel, interim_latent] = blind_deconv(yg, lambda_lmg, lambda_grad, opts);
        processing_time=toc;

        % 保存模糊核估计时间
        processing_data(end+1).image_name = image_name;
        processing_data(end).processing_time = processing_time;

        % Final Deblur:
        y = im2double(I);
        if ~saturation
            Latent = ringing_artifacts_removal(y, kernel, 2e-3, 1e-3, 0);
        else
            Latent = whyte_deconv(y, kernel);
        end
       % k_out = kernel/max(kernel(:));
           k_out=k_rescale(kernel);
        kernel_name = strcat(image_name, '_kernel.png');
        kernel_path = fullfile(kernel_subfolder_path, kernel_name);
        imwrite(k_out, kernel_path);
        deblurImg=im2double(Latent);
        result_name = strcat(image_name, '_result.png');
        result_path = fullfile(result_subfolder_path, result_name);
        imwrite(deblurImg, result_path);
        disp([image_name ' 已完成处理.']);
    end
end
save('./Results/Mixed_results/Chen_et_all/Processing_data.mat', 'processing_data');
rmpath('./Chen_Function')