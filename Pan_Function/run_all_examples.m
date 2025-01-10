clc;
clear;
close all;



opts.I_reg_wt = 5e-3;
opts.gamma_correct = 1.0;
opts.Ik_iter = 4;

fn = 'images/fishes.jpg'; opts.kernel_size = 45;
opts.gamma_correct = 1.0; opts.I_reg_wt = 1e-3;

blurred = im2double(imread(fn));
[deblurImg, kernel] = blind_deconv_main(blurred, opts);
figure; imshow(deblurImg,[]);
kw = kernel - min(kernel(:));
kw = kw./max(kw(:));
imwrite(kw,['results/' fn(7:end-4) '_kernel.png'])
imwrite(deblurImg,['results/' fn(7:end-4) '_out.png'])
