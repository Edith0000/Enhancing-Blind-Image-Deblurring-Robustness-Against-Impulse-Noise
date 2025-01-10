function [img_denoise] = median_filter(img_noise, N)
    [ROW, COL, ~] = size(img_noise);  % ��ȡͼ��ĳߴ磬����ͨ����
    img_noise = im2double(img_noise);
    img_denoise = zeros(ROW, COL, 3);  % ����������ͼ����ͬ��С�Ŀհ׻���

    for k = 1:3  % ����ÿ����ɫͨ��
        for i = 1:ROW - (N-1)
            for j = 1:COL - (N-1)
                mask = img_noise(i:i+(N-1), j:j+(N-1), k);  % ��ȡ��ǰ��ɫͨ�����˲�����
                s = sort(mask(:));
                img_denoise(i+(N-1)/2, j+(N-1)/2, k) = s((N*N+1)/2);  % �Ե�ǰ��ɫͨ��������ֵ�˲�
            end
        end
    end
end