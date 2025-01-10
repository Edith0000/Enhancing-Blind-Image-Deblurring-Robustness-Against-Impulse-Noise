
% Proposed algorithm: Tarmizi Adam & Yin Mingming (7/9/2019)
% L0 + OGS-TV
% code is currently working fine.


function out = L0_OGS_ADMM1_26(f,opts)

lam         = opts.lam; 
o           = opts.O; % mask
Nit         = opts.Nit;
tol         = opts.tol; 
grpSz       = opts.grpSz; %Group size
Nit_inner   = opts.Nit_inner;

beta_1 = 100;
beta_2 = 100;
beta_3 = 100;
beta_4=0;

[row, col]  = size(f);
u_org    = f;





tg = tic;
  
 
     f_bar=ACmF(u_org);%中值滤波
     noise_matrix=u_org-f_bar;
     index_matrix=zeros(size(u_org));
     index_matrix(abs(noise_matrix)>1e-4)=1;
     u = gradient_guidedfilter(f_bar,f_bar,r,0.01);
     v = ones(size(u));

% relError        = zeros(Nit,1);
% psnrGain        = relError;     % PSNR improvement every iteration
% ssimGain        = relError;

epsi_1         = 1e-3*randn(size(f));
epsi_2        = epsi_1;
delta         = epsi_1;
pio          = epsi_1;
org_o=o;

%eigK        = psf2otf(K,[row col]); %In the fourier domain
%eigKtK      = abs(eigK).^2;
eigDtD      = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Dux, Duy] = D(u);
 
% f_bar=zeros(size(eigDtD));
%imshow(f_bar,[])

lhs         = beta_2*ones(size(eigDtD)) + beta_1*eigDtD+beta_4*ones(size(eigDtD)); % From normal eqns.
%Ku__f       = imfilter (u,K,'circular') -f; % Ku-f
Ku__f=u-f;
     
     %u=f_bar;
%      u=f_bar;
%       r=3;
%     
     

   %  end
for k = 1:Nit 
      
    % u = gradient_guidedfilter(f_bar,f_bar,r,0.01);
%        u=u_org;
%      u_old   = u;
     %*** solve y - subproblem ***
     w = v.*o;
      
     A = beta_3*v.*w + beta_2;
     B = -delta - beta_2*Ku__f;
     C = pio.*w;
     
     y = threshold_1(B./A,C./A);
     
     %*** solve x - subproblem (OGSTV problem) ***
     
     
     x1 = gstvdm(Dux + epsi_1/beta_1 , grpSz , lam/beta_1, Nit_inner);
     x2 = gstvdm(Duy + epsi_2/beta_1 , grpSz , lam/beta_1, Nit_inner);
    if k<5
      x1=x1.*index_matrix+Dux.*(1-index_matrix);
      x2=x2.*index_matrix+Duy.*(1-index_matrix);
     end
     %*** solve u - subproblem ***
     ftemp   =  beta_2*y + beta_2*f - delta;
     rhs     = ftemp+ Dt(beta_1*x1 - epsi_1 ,beta_1*x2 - epsi_2);%+beta_4*f_bar;
     u       = fft2(rhs)./lhs;
     u       = real(ifft2(u));
  
  figure(1)
  imshow(u,[])
     % u=u.*index_matrix+u_old.*(1-index_matrix);
  
     
     [Dux, Duy]  = D(u);
     %Ku__f       = imfilter (u,K,'circular') -f;
      Ku__f=u-f;
     %*** solve v - subproblem ***
     s = beta_3*y.*y.*o  ;
     
     c = pio.*abs(y.*o) - 1 ;
     v = max(-c./s,0);
     v = min(v,1);
         
     %*** Update the Lagrange multipliers ***
     epsi_1 = epsi_1 + beta_1*(Dux - x1);
     epsi_2 = epsi_2 + beta_1*(Duy - x2);
      
     delta = delta + beta_2*(Ku__f - y);
     pio   = pio   + beta_3*(v.*abs(y.*o));
      
      
      relError(k)    = norm(u - u_old,'fro')/norm(u, 'fro');
    
      
       if relError(k) < tol
          break;
      end
      
end

tg = toc(tg);

    out.sol                 = u;
    %out.relativeError       = relError(1:k);
%     out.psnrGain            = psnrGain(1:k);
%     out.ssimGain            = ssimGain(1:k);
    out.cpuTime             = tg;
   
    out.OverallItration     = size(out.relativeError,1);

end

function [D,Dt] = defDDt()
D  = @(U) ForwardDiff(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(U)
 Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
 Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DtXY = Dive(X,Y)
  % Transpose of the forward finite difference operator
  % is the divergence fo the forward finite difference operator
  DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
  DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];   
end


function x = threshold_1(a,b)
x = -sign(a).*max(0,abs(a) - b);
end

