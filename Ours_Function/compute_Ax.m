
function y = compute_Ax(x, p)
    x_f = psf2otf(x, p.img_size);
    y = otf2psf((p.m .* x_f), p.psf_size);
    y = y + p.lambda * x;
end