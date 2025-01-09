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
blur_noise_folder = './Data/dif_noise/blur_noise';
kernel_folder = './Results/dif_noise_results/Dong/kernel';
result_folder = './Results/dif_noise_results/Dong/result';

%获取blur_noise_image文件夹中的子文件夹列表
subfolders = dir(blur_noise_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));


% 保存去模糊时间的结构体数组
processing_data=struct('image_name',{},'processing_time',{});



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
        %  opts.kernel_size=kernel_size(kernel_label);
        true_kernel_size=kernel_size(kernel_label);
        
        
        if true_kernel_size<20
            opts.kernel_size=19;
        else
            opts.kernel_size=29;
        end
        
        I=imread(image_path);  % 模糊带噪声图像
        Y_b=im2double(I);
        
        y = Y_b;
        tic
        
        
        [level_kernel_1, latent_1,weight_k1,weight_x1] = blind_deconv_main_ours(Y_b(:,:,1), opts);
        [level_kernel_2, latent_2,weight_k2,weight_x2] = blind_deconv_main_ours(Y_b(:,:,2), opts);
        [level_kernel_3, latent_3,weight_k3,weight_x3] = blind_deconv_main_ours(Y_b(:,:,3), opts);
        
        
        kernel=(level_kernel_1+level_kernel_2+level_kernel_3)./3;
        kernel=kernel./sum(sum(kernel));
        latent(:,:,1)=latent_1;
        latent(:,:,2)=latent_2;
        latent(:,:,3)=latent_3;
        
        
        
        processing_time=toc;
        % 保存模糊核估计时间
        processing_data(end+1).image_name = image_name;
        processing_data(end).processing_time = processing_time;
        
        
        
        
        k_out=k_rescale(kernel);
        kernel_name = strcat(image_name, '_kernel.png');
        kernel_path = fullfile(kernel_subfolder_path, kernel_name);
        imwrite(k_out, kernel_path);
        
        result_name = strcat(image_name, '_result.png');
        result_path = fullfile(result_subfolder_path, result_name);
        imwrite(latent, result_path);
        disp([image_name ' 已完成处理.']);
        clear latent
    end
end
save('./Results/dif_noise_results/Dong/Processing_data.mat', 'processing_data');
rmpath('./Dong_Function')