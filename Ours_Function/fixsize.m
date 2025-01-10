function nf=fixsize(f,nk1,nk2) % 通过删除或补充矩阵的行和列，将矩阵调整为指定的目标尺寸
[k1,k2]=size(f);

while((k1~=nk1)||(k2~=nk2))
    
    if (k1>nk1)
        s=sum(f,2); %每一行进行求和{sum（a，dim）dim=1（列求和）或2（行求和）}
        if (s(1)<s(end))
            f=f(2:end,:); %第一行元素之和小于最后一行，删除第一行
        else
            f=f(1:end-1,:); %第一行元素之和大于最后一行，删除最后一行
        end
    end
    
    if (k1<nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            tf=zeros(k1+1,size(f,2));
            tf(1:k1,:)=f; % 第一行元素之和小于最后一行，在矩阵的顶部添加一行，并填充为零
            f=tf;
        else
            tf=zeros(k1+1,size(f,2));
            tf(2:k1+1,:)=f; % 第一行元素之和大于最后一行，在矩阵的底部添加一行，并填充为零
            f=tf;
        end
    end
    %对列进行与行一样的操作
    if (k2>nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            f=f(:,2:end);
        else
            f=f(:,1:end-1);
        end
    end
    
    if (k2<nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            tf=zeros(size(f,1),k2+1);
            tf(:,1:k2)=f;
            f=tf;
        else
            tf=zeros(size(f,1),k2+1);
            tf(:,2:k2+1)=f;
            f=tf;
        end
    end
    
[k1,k2]=size(f); % 模糊核的尺寸的长和宽

end

nf=f;
end