The Matlab code for "Robust Kernel Estimation with Outliers Handling for Image Deblurring", CVPR2016 submission, Paper ID 883.


How to use:

1. Run "run_all_examples.m" to try the examples included in this package. The estimated results are saved in the "results" folder.
2. Run "run_eccv12_dataset.m" to generate our results on the dataset of Kohler et al., ECCV 2012.
3. Run "run_levin_dataset.m" to generate our results on the dataset of Levin et al., CVPR 2009.
4. Run "run_low_illumination_dataset.m" to generate our results on the prposed low-illumination dataset. 

Important note: 

1. The code is tested in MATLAB R2013b under the MS Windows 7 64bit version with an Intel Core i7-4790CPU@3.60GHz and 28GB RAM. We recommend to run the code using MATLAB R2013b or later version.
2. Please note that the algorithm sometimes may converge to an incorrect result. When you obtain such an incorrect result, please re-try to deblur with a slightly changed parameters(e.g., using large blur kernel sizes or gamma correction (2.2)).
3. As reported in our paper, the current implementation may take several minutes for the blurred image with large resolution and blur kernel size.
4. Due to the size limit, we do not include the datasets of Kohler et al., ECCV 2012 and the proposed low-iilumination dataset in this folder. 
However, our pre-computed results on the datasets of Kohler et al., ECCV 2012 have been included in the folder "eccv12_dataset/results".
The proposed datasets will be made available to the public.



There are a few parameters that need to be specified by users.

Kernel estimation part:
'opts.kernel_size':   the size of blur kernel
'opts.gamma_correct': gamma correction for the input image (typically set as 1 and 2.2)

Non-blind deconvolution part:
"opts.I_reg_wt": regularization weight in the final non-blind deconvolution (i.e.,(15) in the manuscript) [1e-3,1e-2];

======================================================
The descriptions of main functions:
======================================================

@blind_deconv_main: the coarse-to-fine blind deconvolution.
@blind_deconv_level: Alternatively solve the intermediate latent image and kernel at one level.
@deconv_interim_irls: Solve the intermediate latent image by using IRLS method
@conjgrad: Solve the the quadratic form at each IRLS iteration. It is also the sub-function of "deconv_interim_irls.m" and "robust_deconv".
@kernel_irls_solver: Solve the blur kernel by IRLS method.
@conjgrad_for_kernel: the sub-function of "kernel_irls_solver.m", solve the quadratic form of kernel estimation model at each IRLS iteration.
@robust_deconv: Using IRLS to solve the final latent image model.
@gradient_confidence_full: Compute r-map (i.e., (8) in the manuscript)
@shock: Shock filter solver


