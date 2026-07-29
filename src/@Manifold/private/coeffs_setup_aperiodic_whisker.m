function[W_0_1,R_0_1, multi_input] = coeffs_setup_aperiodic_whisker(obj,order)
%% SSM_MULTI Setup for the mutli-index calculation
% This function checks if the symmetries described in $\textit{Explicit Kernel
% Extraction and Proof ofSymmetries of SSM Coefficients - Multi-Indexversion}$
% can be used. This is the  case if the inputs $\texttt{A,B,F}$ are all purely
% real and the eigenvectors are compl. conjugate for complex conjugate eigenvalues
%, we say the system is real. If this is not the case, then the full coefficients
% get calculated and we say the system is not (purely) real. This function is designed to distinguish
% between the case where inherent symmetries are present and the case where there
% are no such symmetries and then prepare to calculate the SSM coefficients.
%
% If the system is real (the symmetries exist) then only the coefficients up
% to the conjugate center index (in conjugate ordering) are calculated.

Lambda_M = sparse(diag(obj.E.spectrum)); % Master Eigenvalues, in order of rev_lex ordering
Lambda_M_vector      = diag(Lambda_M);

Vtrial = obj.System.V;
V_M = Vtrial(:,1:2);          % Left eigenvectors of the modal subspace, order corresponds to rev_lex
W_M = obj.E.adjointBasis;   % Right eigenvectors of the modal subspace

l   = size(Lambda_M,1) + 1; % include epsilon contribution IMP
        z_k = zeros(1,order); 
        for k = 1:order
            z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
        end

%%
        %% No symmetries can be exploited, set up in rev_lex ordering
        switch obj.Options.Solve_Method

            case 'Newmark'
                M = obj.System.M;
                K = obj.System.K;
                C = obj.System.C;
                F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
                F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
                n = size(M,1);
                h = obj.System.dh;
                h = 0.01;
                IC = zeros(n,1);
                q0 = IC;
                qd0 = IC;
                qdd0 = -M\(F_ext(0).');
                tmax = obj.System.tend;
       
                TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005);
                TI_NL.linear = 1;

                % Linear Residual evaluation function handle
                residual = @(q,qd,qdd,t) residual_linear_inhomo( q, qd, qdd, t, M,K,C, F_ext);

                % Time Integration
                
                TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
                

                A_dum = zeros(2*size(TI_NL.Solution.q,1),z_k(1),1,size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,1,1,:) = [TI_NL.Solution.q(:,1:1:end);TI_NL.Solution.qd(:,1:1:end)];
                A_dum(:,2,1,:) = V_M(:,1).*ones(2*size(TI_NL.Solution.q,1),size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,3,1,:) = V_M(:,2).*ones(2*size(TI_NL.Solution.q,1),size(TI_NL.Solution.q(1,1:1:end),2));
               
                W_0_1(1).coeffs = A_dum;
                    
                H{1}  = A_dum;
                A_dum = zeros(l-1,z_k(1),1,size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,1,1,:) = 0*ones(l-1,size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,2,1,:) = Lambda_M(:,1).*ones(l-1,size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,3,1,:) = Lambda_M(:,2).*ones(l-1,size(TI_NL.Solution.q(1,1:1:end),2));
                
                R_0_1(1).coeffs = A_dum;
                % Set up array containing index numbers at every order
%                 l   = 1;
%                 z_k = zeros(1,order); 
%                 for k = 1:order
%                      z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
%                 end  
        
                multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
 
                multi_input.endTime = size(TI_NL.Solution.q(:,1:1:end),2);
                multi_input.ttime = TI_NL.Solution.time(1:1:end);
                multi_input.H = H;
                multi_input.nl_order = numel(obj.System.F);
                multi_input.l = l;
                multi_input.ordering = 'revlex';

            case 'Greens'
                % New leg with history
                 lambda = obj.System.lambda;
%                 Long_tspan=obj.System.Long_span;
%                 Forcing_history = obj.System.Fext(Long_tspan);
% 
%                 n_steps_long =length(Long_tspan)-1;
%                 ord_l = 1;
% 
%                 nt_l = n_steps_long*ord_l+ 1;
%                 dt_l = Long_tspan(end)/(nt_l - 1);
%                 t_l = Long_tspan(1):dt_l:Long_tspan(end);
% 
%                 w_l = NewtonCotes(ord_l); % weights for integration over an interval
%                 w0_l = [w_l(2:end-1) w_l(1)+w_l(end)]; % repeating block in the weights vector
%                 weights_l = dt_l*[w_l(1) repmat(w0_l,1,n_steps_long-1) w_l(2:end)]; % weights for the whole grid
%                 Ft_l  = [-Forcing_history;sparse(obj.System.n,max(size(t_l)))];
%                 phi_l = W' * Ft_l;
%                 Explambda_l = exp(-lambda.*t_l); 
%                 Base_integral = V*sum(weights_l.*(Explambda_l.*phi_l),2);
%                 
                M = obj.System.M;
                K = obj.System.K;
                C = obj.System.C;
                Mi = obj.System.Minv;
                V = obj.System.V;
                W = obj.System.W;
                

%                 F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','spline');
%                 F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
                n_steps =length(obj.System.tforce)-1;
                ord = obj.System.ncc_order;

                nt = n_steps*ord+ 1;
                dt = (obj.System.tforce(end)-obj.System.tforce(1))/(nt - 1);
                t = obj.System.tforce(1):dt:obj.System.tforce(end);
                et = 0:dt:(obj.System.tforce(end)-obj.System.tforce(1));

                w = NewtonCotes(ord); % weights for integration over an interval
                w0 = [w(2:end-1) w(1)+w(end)]; % repeating block in the weights vector
                weights = dt*[w(1) repmat(w0,1,n_steps-1) w(2:end)]; % weights for the whole grid
%                 Ft  = [sparse(obj.System.n,max(size(t))); -Mi*obj.System.Fext(t.')];
                Ft  = [-obj.System.Fext(t);sparse(obj.System.n,max(size(t)))];
                phi = W' * Ft;
                obj.System.Explambda = exp(lambda.*et); 
                eta = zeros(size(phi));
                tic
                for j = 1:2*size(M,1)  
                    eta_d = conv(weights.* phi(j,:),obj.System.Explambda(j,:));
                    eta(j,:) = eta_d(1:max(size(t)));
                    eta(j,1) = 0;
                end
                toc
                z = real(V*eta);
%                 z = eta;
%                 n = size(M,1);
%                 tcalc = obj.System.tforce(:);
%                 tnewc = 0:0.001:obj.System.tforce(end);
%                 tcalc = tnewc.';
%                 
%                 Con_Force = -M\(F_ext(tnewc.').');
% %                 Con_Force = -M\obj.System.Fext;
%                 Con_Force_full = [sparse(obj.System.n,max(size(tcalc)));Con_Force];
%                 Solution_full = sparse(2*(obj.System.n),max(size(tcalc)));
%                 tic 
%                 for indi = 2:max(size(tcalc))
%                     Solution_full(:,indi) = obj.System.Timelin{1,1}*(Solution_full(:,indi-1)+tcalc(2)*Con_Force_full(:,indi-1)/2) + tcalc(2)/2*Con_Force_full(:,indi);
%                 end   
%                 toc
                A_dum = zeros(2*(obj.System.n),z_k(1),1,max(size(t)));
                A_dum(:,1,1,:) = z;
                A_dum(:,2,1,:) = V_M(:,1).*ones(2*(obj.System.n),max(size(t)));
                A_dum(:,3,1,:) = V_M(:,2).*ones(2*(obj.System.n),max(size(t)));
               
                W_0_1(1).coeffs = A_dum;
                
                

                H{1}  = A_dum;
        
                % Set up array containing index numbers at every order
%                 l   = 1;
%                 z_k = zeros(1,order); 
%                 for k = 1:order
%                      z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
%                 end  
        
                multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
                multi_input.dt = dt;
                multi_input.weights = weights;
                
                multi_input.endTime = max(size(t));
                multi_input.ttime = t;
                multi_input.etime = et;
                multi_input.H = H;
                multi_input.nl_order = numel(obj.System.F);
                multi_input.l = l;
                multi_input.ordering = 'revlex';

          case 'Greens_Gauss'
              lambda = obj.System.lambda;
              M = obj.System.M;
              V = obj.System.V;
              W = obj.System.W;
              F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
              F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              
              Ng = 25;
              [xG,wG]=lgwt(Ng,0,obj.System.tforce(2)-obj.System.tforce(1));
              tG_new = flip(xG) + obj.System.tforce ;
              TGnew = tG_new(:);
              TGnew = TGnew(1:end-Ng).';
              etG = obj.System.tforce - obj.System.tforce(1);
              t_minus_s = (obj.System.tforce(2)-obj.System.tforce(1))-flip(xG);
              etG_minus = flip(t_minus_s) + etG;
              etGM= etG_minus(:);
              etGM = etGM(1:end-Ng).';
              WG = repmat(wG.',1,max(size(obj.System.tforce))-1);
              FtG  = [-F_ext(TGnew.').';sparse(obj.System.n,max(size(TGnew)))];
              phiG = W'*FtG;% (rearrange IMP)
              GreenPos = exp(lambda.*etGM); 
%               GreenPos = GreenPos(find(GreenPos~=0)); 
              obj.System.Explambda = GreenPos;
              eta = zeros(2*size(M,1),max(size(obj.System.tforce)));
              for j = 1:2*size(M,1)  
                    eta_d = conv(WG.*phiG(j,:),obj.System.Explambda(j,find(obj.System.Explambda(j,:)~=0)));
                    ETA_M = eta_d(Ng:Ng:max(size(TGnew)));
                    eta(j,:) = [0,ETA_M];
              end
           
                
               z = real(V*eta);

               A_dum = zeros(2*(obj.System.n),z_k(1),1,max(size(obj.System.tforce)));
               A_dum(:,1,1,:) = z;
               A_dum(:,2,1,:) = V_M(:,1).*ones(2*(obj.System.n),max(size(obj.System.tforce)));
               A_dum(:,3,1,:) = V_M(:,2).*ones(2*(obj.System.n),max(size(obj.System.tforce)));
               
               W_0_1(1).coeffs = A_dum;
                
                

                H{1}  = A_dum;
               
        
               % Set up array containing index numbers at every order
%                l   = 1;
%                z_k = zeros(1,order); 
%                for k = 1:order
%                     z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
%                end  
%         
               multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
               multi_input.dt = obj.System.tforce(2);
               multi_input.weights = WG;
                
               multi_input.endTime = max(size(obj.System.tforce));
               multi_input.ttime = obj.System.tforce;
               multi_input.etime = etGM;
               multi_input.TTtime = TGnew;
               multi_input.Ng = Ng;
               multi_input.H = H;
               multi_input.nl_order = numel(obj.System.F);
               multi_input.l = l;
               multi_input.ordering = 'revlex';

           case 'Piecewise_exact'
              lambda = obj.System.lambda;
              M = obj.System.M;
              V = obj.System.V;
              W = obj.System.W;
              F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
              F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              
              h = obj.System.tforce(2)-obj.System.tforce(1);
              tnew = 0:0.01:obj.System.tforce(end);
              h = 0.01;
%               Ft  = [-obj.System.Fext;sparse(obj.System.n,max(size(obj.System.tforce)))];
              
              Ft  = [-F_ext(tnew.').';sparse(obj.System.n,max(size(tnew)))];
          
              phi = W' * Ft;
              Ftn = phi(:,1:end-1);
              Ftn_p1 = phi(:,2:end);
              ExpoA = (exp(lambda*h)-1)./(lambda);
              ExpoB = (exp(lambda*h))./(lambda.^2) - 1./(lambda.^2) - h./(lambda);
              PHI = ExpoA.*Ftn + ExpoB.*(Ftn_p1-Ftn)/h;
              et = 0:h:(obj.System.tforce(end)-obj.System.tforce(1));
              obj.System.Explambda = exp(lambda.*et(1:end-1));
              
              multi_input.ExpoA = ExpoA;
              multi_input.ExpoB = ExpoB;
              multi_input.h = h;
%               eta = zeros(2*size(M,1),max(size(obj.System.tforce)));
              eta = zeros(2*size(M,1),max(size(tnew)));
              for j = 1:2*size(M,1)  
                    eta_d = conv(PHI(j,:),obj.System.Explambda(j,:));
                    ETA_M = eta_d(1:max(size(et(1:end-1))));
                    eta(j,:) = [0,ETA_M];
              end
           
                
               z = real(V*eta);

%                A_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
%                H_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
               A_dum = zeros(2*(obj.System.n),z_k(1),1,max(size(tnew)));
               A_dum(:,1,1,:) = z;
               A_dum(:,2,1,:) = V_M(:,1).*ones(2*(obj.System.n),max(size(tnew)));
               A_dum(:,3,1,:) = V_M(:,2).*ones(2*(obj.System.n),max(size(tnew)));
               
               W_0_1(1).coeffs = A_dum;
                
                

                H{1}  = A_dum;
                A_dum = zeros(l-1,z_k(1),1,max(size(tnew)));
                A_dum(:,1,1,:) = 0*ones(l-1,max(size(tnew)));
                A_dum(:,2,1,:) = Lambda_M(:,1).*ones(l-1,max(size(tnew)));
                A_dum(:,3,1,:) = Lambda_M(:,2).*ones(l-1,max(size(tnew)));
                
                R_0_1(1).coeffs = A_dum;
             
        
               % Set up array containing index numbers at every order
%                l   = 1;
%                z_k = zeros(1,order); 
%                for k = 1:order
%                     z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
%                end  

        
               multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
               multi_input.dt = obj.System.tforce(2);
            
                
               multi_input.endTime = max(size(tnew));
               multi_input.Lambda_M_vector = Lambda_M_vector;
%                multi_input.ttime = obj.System.tforce;
               multi_input.ttime = tnew;
               multi_input.etime = et;
               multi_input.H = H;
               multi_input.nl_order = numel(obj.System.F);
               multi_input.l = l;
               multi_input.ordering = 'revlex';

            case 'Piecewise_Exact_DiffParam_Choice'
             
              h = obj.System.tforce(2)-obj.System.tforce(1);
              h = 0.01;
              tnew = 0:h:obj.System.tforce(end);
              M = obj.System.M;
              V = obj.System.V;
              W = obj.System.W;

              F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
              F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              
              
%               Ft  = [-obj.System.Fext;sparse(obj.System.n,max(size(obj.System.tforce)))];
              
              Ft  = [-F_ext(tnew.').';sparse(obj.System.n,max(size(tnew)))];
          
              phi = W' * Ft;

              A_dum = zeros(l-1,z_k(1),1,max(size(tnew)));
              
              A_dum(:,1,1,:) = phi(1:l-1,:);

              A_dum(:,2,1,:) = Lambda_M(:,1).*ones(l-1,max(size(tnew)));
              A_dum(:,3,1,:) = Lambda_M(:,2).*ones(l-1,max(size(tnew)));
                
              R_0_1(1).coeffs = A_dum;  
              lambda = obj.System.lambda;
              
              
              ExpoA = (exp(lambda*h)-1)./(lambda);
              ExpoB = (exp(lambda*h))./(lambda.^2) - 1./(lambda.^2) - h./(lambda);
              ExpoA(1:l-1) = 0;
              ExpoB(1:l-1) = 0;

              ExpoT = exp(lambda*h);
              ExpoT(1:l-1) = 0;
               
              multi_input.ExpoT = ExpoT;
              multi_input.ExpoFn = ExpoA-ExpoB/h;
              multi_input.ExpoFn_1 = ExpoB/h;
              eta = zeros(2*size(M,1),max(size(tnew)));
              for indk = 2:max(size(tnew))
                        eta(:,indk) = (multi_input.ExpoT).*eta(:,indk-1) + (multi_input.ExpoFn_1).*phi(:,indk) + (multi_input.ExpoFn).*phi(:,indk-1);
              end   
                          
              z = real(V*eta);

%                A_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
%                H_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
              A_dum = zeros(2*(obj.System.n),z_k(1),1,max(size(tnew)));
              A_dum(:,1,1,:) = z;
              A_dum(:,2,1,:) = V_M(:,1).*ones(2*(obj.System.n),max(size(tnew)));
              A_dum(:,3,1,:) = V_M(:,2).*ones(2*(obj.System.n),max(size(tnew)));
               
              W_0_1(1).coeffs = A_dum;
                
                

              H{1}  = A_dum;
              
             
        
               % Set up array containing index numbers at every order
%                l   = 1;
%                z_k = zeros(1,order); 
%                for k = 1:order
%                     z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
%                end  

        
               multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
               multi_input.dt = obj.System.tforce(2);
            
                
               multi_input.endTime = max(size(tnew));
               multi_input.Lambda_M_vector = Lambda_M_vector;
%                multi_input.ttime = obj.System.tforce;
               multi_input.ttime = tnew;
               
               multi_input.H = H;
               multi_input.nl_order = numel(obj.System.F);
               multi_input.l = l;
               multi_input.ordering = 'revlex';
             
        end        
end

function [conjugate] = conjugate_ordering(max_order,l_r,l_i)
% Creates conjugate ordered set indices of multi-indices up to order
% max_order. Conjugate ordering is defined in $\textit{Explicit Kernel
% Extraction and Proof ofSymmetries of SSM Coefficients - Multi-Indexversion}$

% Input:
% max_order   highest order of multi-indices that the conjugate ordering is 
%              wanted for
% l_r number of real eigenvalues in the master subspace
% l_i number of imaignary pairs of eigenvalues in the master subspace

% Output
% for a conjugately ordered set Z, rev. lex ordered set K:
% lex2conj index array for construction.   - K(:,lex2conj) =Z
% conj2lex index array for reconstruction. - Z(:,conj2lex) = K
%     i.e if Z_idx(f) = i, then k_f in K(:,f) is in position i in Z
% cci_k - conjugate center index of the set


Z_cci    = zeros(1,max_order); % conjugate center index array
revlex2conj = cell(1,max_order);  % index sets converting lex set to conj set
conj2revlex = cell(1,max_order);  % index sets converting conj set to lex set

for k = 1:max_order
    I = conjugate_flip(l_i,l_r);
    % Multi_indices in reverse lexicographical ordering
    K = flip(sortrows(nsumk(l_r+2*l_i,k,'nonnegative')).',2);
    Y = K(I,:);
    Exempt      = all(K-Y==0);
    Y(:,Exempt) = [];
    
    %Put m,m_c next to each other
    Z          = [K(:,~Exempt);Y];
    Z          = reshape(Z, size(K,1),[]);
    
    %index out all the combos of m, bar(m) that appear twice
    [~,Z_ia,~] = unique(Z.','rows');
    [~,Z_Ia]   = sort(Z_ia);    
    Z     = Z(:,Z_ia(Z_Ia));
    
    % sort the remaining multi-indices
    idx_1 = 1:2:size(Y,2);
    idx_2 = size(Y,2)-idx_1+1;
    Z     = [Z(:,idx_1),K(:,Exempt),Z(:,idx_2)];
    
    % conjugate center index
    cci_k = sum(Exempt,2) + length(idx_1); 

    % index arrays
    [~,~,conj2lex_k] = intersect(K.',Z.','rows','stable');
    [~,~,lex2conj_k] = intersect(Z.',K.','rows','stable');
    
    conj2revlex{k} = conj2lex_k.';
    revlex2conj{k} = lex2conj_k.';
    Z_cci(k)    = cci_k;
end

conjugate.conj2revlex = conj2revlex;
conjugate.revlex2conj = revlex2conj;
conjugate.Z_cci    = Z_cci;
conjugate.l_i      = l_i;
conjugate.l_r      = l_r;
end

function [idx] = conjugate_flip(l_i,l_r)
% This function computes and index array that flips the conjugate coordinate directions  
% Conjugate coordinate directions are defined in $\textit{Explicit Kernel
% Extraction and Proof ofSymmetries of SSM Coefficients - Multi-Indexversion}$

idx = reshape(1:2*l_i,2,[]);
idx = reshape(flip(idx),1,[]);
idx = [idx,2*l_i+1:(2*l_i+l_r)];
end