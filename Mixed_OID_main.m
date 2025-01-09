clc;
clear;
close all; 

addpath(genpath('./OID_code/'));

opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;
%opts.kernel_size = 27;
saturation = 0;
lambda_lmg =4e-3; lambda_grad =4e-3;opts.gamma_correct = 1;
lambda_tv = 0.001; lambda_l0 = 5e-4; weight_ring = 1;

% 设置文件夹路径
blur_noise_folder = './Data/Mixed_noise/blur_noise';
kernel_folder = './Results/Mixed_noise_results/OID/kernel';
result_folder = './Results/Mixed_noise_results/OID/result';
true_result_folder = './Results/Mixed_noise_results_true/OID/result';
clear_image_folder = './Data/Mixed_noise_data/image';

   % 获取清晰图像的图像列表
  clear_image_list = dir(fullfile(clear_image_folder, '*.png'));
  clear_image_list = {clear_image_list.name};
%获取blur_noise_image文件夹中的子文件夹列表
subfolders = dir(blur_noise_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));


% 保存去模糊时间的结构体数组
processing_data=struct('image_name',{},'processing_time',{},'error_rate',{});

% 加载真实的模糊核并储存到一个结构体里
True_kernel_struct = struct();
for k = 1:8
    filename = fullfile('kernel', sprintf('kernel_%d.mat', k));
    data = load(filename);
    variable_name = sprintf('kernel_%d', k);
    True_kernel_struct.(variable_name) = data.(variable_name);
end


opts.prescale = 0; %%downsampling
opts.xk_iter = 4; %% the iterations
opts.last_iter = 4; %% larger if image is very noisy
opts.k_thresh = 20;

opts.isnoisy = 1; %% filter the input for coarser scales before deblurring 0 or 1
%opts.kernel_size = 27;  %% kernel size
opts.predeblur = 'L0';  %% deblurring method for coarser scales; Lp or L0
lambda_grad = 4e-3; %% range(1e-3 - 2e-2)
opts.gamma_correct = 1.0; %% range (1.0-2.2)



%%


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
    
      % 创建 true_result文件夹中对应子文件夹
    true_result_subfolder_path = fullfile(true_result_folder, subfolder_name);
    if ~isfolder(true_result_subfolder_path)
        mkdir(true_result_subfolder_path);
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
        true_kernel_size=kernel_size(kernel_label);
        
      
        if true_kernel_size<20
            opts.kernel_size=19;
        else
            opts.kernel_size=29;
        end

        kernel_name=sprintf('kernel_%d', kernel_label);
        true_kernel=True_kernel_struct.(kernel_name); % 真实模糊核
        true_kernel = k_rescale(true_kernel);
        true_kernel_2=imresize(true_kernel,[opts.kernel_size opts.kernel_size]);
        true_kernel_3=true_kernel_2./sum(sum(true_kernel_2));
       
        
        I=imread(image_path);  % 模糊带噪声图像
        if numel(size(I))>2
            Y_b=im2double(rgb2gray(I));
        else
            Y_b=im2double(I);
        end

        yg = Y_b;
        
        

        tic;
       % [kernel, interim_latent] = blind_deconv(yg, lambda_lmg, lambda_grad, opts);
        
        [kernel, interim_latent] = blind_deconv(yg, lambda_grad, opts);

        processing_time=toc;
        
        % 保存模糊核估计时间
        processing_data(end+1).image_name = image_name;
        processing_data(end).processing_time = processing_time;

        % Final Deblur:
        y = im2double(I);
        Latent = image_estimate(y, kernel, 0.003,0);
        Latent=im2double(Latent);
      
        
        
         %%%真实模糊核 参数设置错误？是否要变成double类型，原先是single
        %Latent_2 =image_estimate(y,true_kernel_3, 0.003,0,opts,true_img);
        Latent_2 =image_estimate(y,true_kernel_3, 0.003,0);
        Latent_2=im2double(Latent_2);
       
    

        
        % 保存模糊核和去模糊结果
        k_out=k_rescale(kernel);
        
%         kernel_name = strcat(image_name, '_kernel.png'); % 根据需要调整保存的文件格式
%         kernel_path = fullfile(kernel_subfolder_path, kernel_name);
%         imwrite(k_out, kernel_path);
%         result_name = strcat(image_name, '_result.png');
%         result_path = fullfile(result_subfolder_path, result_name);
%         imwrite(Latent, result_path);
%         latent_image=imread(result_path);
        % 保存用真实模糊核去模糊的结果
 
        k_out=k_rescale(kernel);
        kernel_name = strcat(image_name, '_kernel.png');
        kernel_path = fullfile(kernel_subfolder_path, kernel_name);
        imwrite(k_out, kernel_path);
        
        result_name = strcat(image_name, '_result.png');
        result_path = fullfile(result_subfolder_path, result_name);
        imwrite(Latent, result_path);
        true_result_name = strcat(image_name, '_true_kernel_image.png');
        true_result_path = fullfile(true_result_subfolder_path, true_result_name);
        imwrite(Latent_2, true_result_path);
        
        latent2_image=imread(true_result_path);
        %计算并储存误差率
        Latent=imread(result_path);
        
        
        clear_image_path=fullfile(clear_image_folder, clear_image_list{j});
        clear_image=imread(clear_image_path);
        clear_image=im2double(clear_image);
        Latent=im2double(Latent);
        
        for tt=1:size(clear_image,3)
            err1(tt)=norm(clear_image(:,:,tt)-Latent(:,:,tt),'fro')/norm(clear_image(:,:,tt)-Latent_2(:,:,tt),'fro');
        end
        error_rate=mean(err1);

       processing_data(end).error_rate = error_rate;
        
        
        
        disp([image_name ' 已完成处理.']);
        clear 
        
    end
end

%%%



