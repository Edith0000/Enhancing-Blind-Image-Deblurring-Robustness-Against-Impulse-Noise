clc;
clear;
close all;

addpath(genpath('./Ours_Function')); % 添加函数路径
addpath(genpath('./Data'));

opts.prescale = 0; %% downsampling 采样频率参数,用于图像下采样
opts.xk_iter = 4; %% the iterations 迭代次数
opts.last_iter = 4; %% 如果图像噪声很大，则迭代次数要增加
opts.k_thresh = 20; %% k的阈值
opts.isnoisy = 1; %% filter the input for coarser scales before deblurring 0 or 1
opts.predeblur = 'L0';  %% deblurring method for coarser scales; Lp or L0
opts.mu=0.01;
opts.lambda=0.05; 
opts.rho_mu=1.15;
opts.rho_lambda=1; 
opts.t1=0.1;
opts.t2=0.1;
opts.sigma=0.11*sqrt(2); %%%  参数1
opts.threshold=0.05; %%% 参数2
opts.band=0.005; %%% 参数3
% opts.alpha1=2e-3;
% opts.beta1=2e-3;
opts.tol=1e-2;

opts.alpha1=1.8e-3; 
opts.beta1=0.0013;
%opts.beta1=0.5*abs(opts.alpha1/log(0.5))
% 设置文件夹路径
blur_noise_folder = './Data/dif_noise/blur_noise';
kernel_folder = './Results/dif_noise_results/Ours/kernel';
result_folder = './Results/dif_noise_results/Ours/result';
true_result_folder = './Results/dif_noise_results/Ours/result';

% 获取blur_noise_image文件夹中的子文件夹列表
subfolders = dir(blur_noise_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

% 保存去模糊时间的结构体数组
processing_times = struct('image_name',{},'processing_time',{});

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

    % 获取子文件夹中的图像列表
    B_N_files = dir(fullfile(subfolder_path, '*.png'));
    B_N_files = {B_N_files.name};

    % 遍历子文件夹中的每张图像
    for j = 1:length(B_N_files)
%     for j = 1:1
        image_name = B_N_files{j};
        image_path = fullfile(subfolder_path, image_name);
        
        kernel_label=ceil(i/3);
        kernel_size=[19 17 15 27 13 21 23 23];
        true_kernel_size=kernel_size(kernel_label);

        % if true_kernel_size<20
        %     opts.kernel_size=19;
        % else
        %     opts.kernel_size=29;
        % end
        opts.kernel_size=true_kernel_size;

        I=imread(image_path);  % 模糊带噪声图像

        if numel(size(I))>2
            Y_b=im2double(rgb2gray(I));
        else
            Y_b=im2double(I);
        end
        [~,noise_level]=AMF2(Y_b,0);  % 噪声水平
        opts.noise_level=noise_level;
        k_estimate_size=opts.kernel_size; % set approximate kernel size
        show_intermediate=true; % show intermediate output or not
        border=20;

        % 盲去模糊过程
        tic;
        [k_estimate,Y_intermediate]=deblur_denoise_main(Y_b(border+1:end-border,border+1:end-border),k_estimate_size,opts);
        processing_time=toc;

        processing_times(end+1).image_name = image_name;
        processing_times(end).processing_time = processing_time;

        % 非盲过程
        I_b = im2double(I); % 彩图
        [I_FHLP] =image_estimate_main(I_b, k_estimate, 0.003,0,opts);
        
        % 保存模糊核和去模糊结果
        k_out=k_rescale(k_estimate);
        kernel_name = strcat(image_name, '_kernel.png'); % 根据需要调整保存的文件格式
        kernel_path = fullfile(kernel_subfolder_path, kernel_name);
        imwrite(k_out, kernel_path);
        result_name = strcat(image_name, '_result.png');
        result_path = fullfile(result_subfolder_path, result_name);
        imwrite(I_FHLP, result_path);        
        
        disp([image_name ' 已完成处理.']);
    end
end
save('./Results/dif_noise_results/Ours/Blind_processing_times.mat', 'processing_times');
rmpath('./Ours_Function');