function threshold=wave_computation(A)
% if size(y_org,3)>1
% y_org=im2double(rgb2gray(y_org));
% else
% y_org=im2double(y_org);
% end
        %A=medfilt2(A);
        
%         A=y_org;
         [cA,cH,cV,cD]=dwt2(A,'haar');
%         figure(2)
%         subplot(1,3,1)
%         %imshow(cH,[])
%         hist(cH(:))
%         subplot(1,3,2)
%         %imshow(cV,[])
%         hist(cV(:))
%         subplot(1,3,3)
%         %imshow(cH,[])
%         hist(cD(:))
        
        
        
        band(1)=length(find(abs(cH(:))>0.3))/length(cH(:));
        band(2)=length(find(abs(cV(:))>0.3))/length(cV(:));
        band(3)=length(find(abs(cD(:))>0.3))/length(cD(:));
        threshold=norm(band,'Inf');
        
end