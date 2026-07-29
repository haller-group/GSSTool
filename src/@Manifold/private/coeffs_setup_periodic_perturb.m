function[A_0_1, multi_input] = coeffs_setup_periodic_perturb(obj,order)
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


%%
        %% No symmetries can be exploited, set up in rev_lex ordering
        switch obj.Options.Solve_Method

            case 'Newmark'
                M = obj.System.M;
                K = obj.System.K;
                C = obj.System.C;
                G_t = obj.System.G_t;
                Omi = obj.System.omi;
                
                n         = size(G_t, 1);
                N_period  = size(G_t, 3);
                T_period  = 2*pi / Omi;
                dt        = T_period / N_period;                     % step size
                t_periodic = (0 : N_period-1) * dt;                 % 1 × N_period, [0, T)
                
                % Flatten G_t to (N_period × n²) for griddedInterpolant
                %   row t, col (i-1)*n+j  =  G_{ij}(t)
                G_flat = reshape(permute(G_t, [3 1 2]), N_period, n*n);  % N_period × n²

                % One interpolant per (i,j) entry, built on the periodic grid
                G_interp = griddedInterpolant(t_periodic(:), G_flat, 'linear');

                % Function handle: scalar or vector t  →  n × n matrix
                %   mod(t, T_period) wraps any t into [0, T)
                %   eps*T_period guard keeps mod(T, T)=0 exactly inside the grid
                G_ext = @(t) reshape( ...
                G_interp(mod(t, T_period - eps*T_period)), ...
                            n, n);

                F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
                F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
                n = size(M,1);
                h = obj.System.dh;
                
                IC = zeros(n,1);
                q0 = IC;
                qd0 = IC;
                qdd0 = -M\(F_ext(0).');
                tmax = obj.System.tend;
       
                TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005);
                TI_NL.linear = 1;

                % Linear Residual evaluation function handle
                residual = @(q,qd,qdd,t) residual_linear_inhomo_t( q, qd, qdd, t, M,K,C, F_ext,G_ext);

                % Time Integration
                
                TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
                

                A_dum = zeros(2*size(TI_NL.Solution.q,1),1,1,size(TI_NL.Solution.q(1,1:1:end),2));
                H_dum = zeros(2*size(TI_NL.Solution.q,1),1,1,size(TI_NL.Solution.q(1,1:1:end),2));
                A_dum(:,1,1,:) = [TI_NL.Solution.q(:,1:1:end);TI_NL.Solution.qd(:,1:1:end)];
                H_dum(:,1,1,:) = [TI_NL.Solution.q(:,1:1:end);TI_NL.Solution.qd(:,1:1:end)];
                A_0_1(1).coeffs = A_dum;

                H{1}  = H_dum;
        
                % Set up array containing index numbers at every order
                l   = 1;
                z_k = zeros(1,order); 
                for k = 1:order
                     z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
                end  
        
                multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
 
                multi_input.endTime = size(TI_NL.Solution.q(:,1:1:end),2);
                multi_input.ttime = TI_NL.Solution.time(1:1:end);
                multi_input.H = H;
                multi_input.G_ext = G_ext;
                multi_input.t_periodic = t_periodic;
                multi_input.nl_order = numel(obj.System.F_shifted);
                multi_input.l = 1;
                multi_input.ordering = 'revlex';
                multi_input.time_no = 0;

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
                A_dum = zeros(2*(obj.System.n),1,1,max(size(t)));
                H_dum = zeros(2*(obj.System.n),1,1,max(size(t)));
                A_dum(:,1,1,:) = z;
                H_dum(:,1,1,:) = z;
                A_0_1(1).coeffs = A_dum;

                H{1}  = H_dum;
        
                % Set up array containing index numbers at every order
                l   = 1;
                z_k = zeros(1,order); 
                for k = 1:order
                     z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
                end  
        
                multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
                multi_input.dt = dt;
                multi_input.weights = weights;
                
                multi_input.endTime = max(size(t));
                multi_input.ttime = t;
                multi_input.etime = et;
                multi_input.H = H;
                multi_input.nl_order = numel(obj.System.F);
                multi_input.l = 1;
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

               A_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
               H_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
               A_dum(:,1,1,:) = z;
               H_dum(:,1,1,:) = z;
               A_0_1(1).coeffs = A_dum;

               H{1}  = H_dum;
        
               % Set up array containing index numbers at every order
               l   = 1;
               z_k = zeros(1,order); 
               for k = 1:order
                    z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
               end  
        
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
               multi_input.l = 1;
               multi_input.ordering = 'revlex';

           case 'Piecewise_exact'
              lambda = obj.System.lambda;
              M = obj.System.M;
              V = obj.System.V;
              W = obj.System.W;
              F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
              F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              h = obj.System.dh; 
              tnew = 0:h:obj.System.tforce(end);
%               h = obj.System.tforce(2)-obj.System.tforce(1);
%               tnew = 0:0.001:obj.System.tforce(end);
%               h = 0.001;
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
               A_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
               H_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
              
               A_dum(:,1,1,:) = z;
               H_dum(:,1,1,:) = z;
               A_0_1(1).coeffs = A_dum;

               H{1}  = H_dum;
        
               % Set up array containing index numbers at every order
               l   = 1;
               z_k = zeros(1,order); 
               for k = 1:order
                    z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
               end  

        
               multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
               multi_input.dt = obj.System.tforce(2);
            
                
               multi_input.endTime = max(size(tnew));
%                multi_input.ttime = obj.System.tforce;
               multi_input.ttime = tnew;
               multi_input.etime = et;
               multi_input.H = H;
               multi_input.nl_order = numel(obj.System.F);
               multi_input.l = 1;
               multi_input.ordering = 'revlex';
             
      
            case 'Piecewise_exact_first_order'
              lambda = obj.System.lambda;
              A = obj.System.A;
              V = obj.System.V;
              W = obj.System.W;
              F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
              F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              h = obj.System.dh; 
              tnew = 0:h:obj.System.tforce(end);
%               h = obj.System.tforce(2)-obj.System.tforce(1);
%               tnew = 0:0.001:obj.System.tforce(end);
%               h = 0.001;
%               Ft  = [-obj.System.Fext;sparse(obj.System.n,max(size(obj.System.tforce)))];
              
              Ft  = -F_ext(tnew.').';
          
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
              eta = zeros(size(A,1),max(size(tnew)));
              for j = 1:size(A,1)  
                    eta_d = conv(PHI(j,:),obj.System.Explambda(j,:));
                    ETA_M = eta_d(1:max(size(et(1:end-1))));
                    eta(j,:) = [0,ETA_M];
              end
           
                
               z = (V*eta);

%                A_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
%                H_dum = zeros(2*(obj.System.n),1,1,max(size(obj.System.tforce)));
               A_dum = zeros(size(A,1),1,1,max(size(tnew)));
               H_dum = zeros(size(A,1),1,1,max(size(tnew)));
              
               A_dum(:,1,1,:) = z;
               H_dum(:,1,1,:) = z;
               A_0_1(1).coeffs = A_dum;

               H{1}  = H_dum;
        
               % Set up array containing index numbers at every order
               l   = 1;
               z_k = zeros(1,order); 
               for k = 1:order
                    z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
               end  

        
               multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
               multi_input.dt = obj.System.tforce(2);
            
                
               multi_input.endTime = max(size(tnew));
%                multi_input.ttime = obj.System.tforce;
               multi_input.ttime = tnew;
               multi_input.etime = et;
               multi_input.H = H;
               multi_input.nl_order = numel(obj.System.F);
               multi_input.l = 1;
               multi_input.ordering = 'revlex';
             
        case 'Second_order_piecewise_exact'
              
              ticstarto = tic;
              M = obj.System.M;
              n = length(M);
              K = obj.System.K;
              C = obj.System.C;
              h = obj.System.dh;

              if obj.System.nmodes
                 [VV, dd] = eigs(full(K),full(M), obj.System.nmodes,'smallestabs');
              else
                 [VV, dd] = eig(full(K),full(M));
              end   
               
              dd = diag(dd);
              [~, ind] = sort(dd);
              V = VV(:,ind);
                
              mu = diag(V.' * M * V);
              U = V * diag( 1./ sqrt(mu) ); 
              omega0 = sqrt(diag((U.' * K * U)));
              zeta = diag((U.' * C * U))./ (2*omega0);

              lambda_p = (-zeta + sqrt(zeta.^2 - 1)).* omega0;
              lambda_m = (-zeta - sqrt(zeta.^2 - 1)).* omega0;
              cd = find(zeta==1); 
              oud = find(zeta>0 & zeta~=1);

              oudu = find(zeta<0 & zeta~=-1); %% for future
              cdu = find(zeta==-1); %% for future
              multi_input.cd = cd;
              multi_input.oud = oud;
              Am =lambda_p.*0;
              Bm =lambda_p.*0;
              Cm =lambda_p.*0;
              Dm =lambda_p.*0;
              
              Amp =lambda_p.*0;
              Bmp =lambda_p.*0;
              Cmp =lambda_p.*0;
              Dmp =lambda_p.*0;

              
                
              if isempty(oud)==0
                Am(oud) = exp(lambda_p(oud)*h).*(lambda_m(oud)-lambda_p(oud).*exp((lambda_m(oud)-lambda_p(oud))*h))./(lambda_m(oud)-lambda_p(oud));
                Bm(oud) = exp(lambda_p(oud)*h).*(exp((lambda_m(oud)-lambda_p(oud))*h)-1)./(lambda_m(oud)-lambda_p(oud));
                Cm(oud) = 1./(lambda_m(oud).*lambda_p(oud)) + (lambda_m(oud)+lambda_p(oud))./((lambda_m(oud).*lambda_p(oud)).^2*h) +  ...
                  exp(lambda_p(oud)*h)/h.*(exp((lambda_m(oud)-lambda_p(oud))*h)./((lambda_m(oud)-lambda_p(oud)).*lambda_m(oud).^2)-1./((lambda_m(oud)-lambda_p(oud)).*lambda_p(oud).^2));

                Dm(oud) = exp(lambda_p(oud)*h).*(lambda_p(oud).*exp((lambda_m(oud)-lambda_p(oud))*h)-lambda_m(oud))./((lambda_m(oud)-lambda_p(oud)).*lambda_p(oud).*lambda_m(oud)) - (lambda_m(oud)+lambda_p(oud))./((lambda_m(oud).*lambda_p(oud)).^2*h) -  ...
                  exp(lambda_p(oud)*h)/h.*(exp((lambda_m(oud)-lambda_p(oud))*h)./((lambda_m(oud)-lambda_p(oud)).*lambda_m(oud).^2)-1./((lambda_m(oud)-lambda_p(oud)).*lambda_p(oud).^2));

                Amp(oud) = exp(lambda_p(oud)*h).*lambda_p(oud).*lambda_m(oud)./(lambda_m(oud)-lambda_p(oud)).*(1-exp((lambda_m(oud)-lambda_p(oud))*h));

                Bmp(oud) = exp(lambda_p(oud)*h)./(lambda_m(oud)-lambda_p(oud)).*(lambda_m(oud).*exp((lambda_m(oud)-lambda_p(oud))*h)-lambda_p(oud));

                Cmp(oud) = exp(lambda_p(oud)*h)/h.*(exp((lambda_m(oud)-lambda_p(oud))*h).*1./(lambda_m(oud).*(lambda_m(oud)-lambda_p(oud)))-1./(lambda_p(oud).*(lambda_m(oud)-lambda_p(oud)))) + 1./(h*lambda_m(oud).*lambda_p(oud)) ;

                Dmp(oud) = exp(lambda_p(oud)*h)./((lambda_m(oud)-lambda_p(oud))).*(exp((lambda_m(oud)-lambda_p(oud))*h)-1) - Cmp(oud);

              end

              if isempty(cd)==0
                  Am(cd) = -exp(lambda_p(cd)*h).*lambda_p(cd)*h;
                  Bm(cd) = exp(lambda_p(cd)*h)*h;
                  Cm(cd) = exp(lambda_p(cd)*h)./(lambda_p(cd).^2) + 1./(lambda_p(cd).*lambda_m(cd)) + (lambda_m(cd)+lambda_p(cd))./((lambda_p(cd).*lambda_m(cd)).^2*h);
                  Dm(cd) = exp(lambda_p(cd)*h)./lambda_p(cd)*h -(lambda_m(cd)+lambda_p(cd))./((lambda_p(cd).*lambda_m(cd)).^2*h)-exp(lambda_p(cd)*h)./(lambda_p(cd).^2);

                  Amp(cd) = -exp(lambda_p(cd)*h).*lambda_p(cd).*lambda_m(cd)*h;
                  Bmp(cd) = exp(lambda_p(cd)*h).*lambda_p(cd)*h;
                  Cmp(cd) = exp(lambda_p(cd)*h)./lambda_p(cd)+1./(lambda_p(cd).*lambda_m(cd)*h);
                  Dmp(cd) = exp(lambda_p(cd)*h)*h - 1./(h*lambda_p(cd).^2) -exp(lambda_p(cd)*h)./(lambda_p(cd));

              end   
              time_no = toc(ticstarto);
              multi_input.time_no = time_no;
              multi_input.Am = real(Am);
              multi_input.Bm = real(Bm);
              multi_input.Cm = real(Cm);
              multi_input.Dm = real(Dm);

              multi_input.Amp = real(Amp);
              multi_input.Bmp = real(Bmp);
              multi_input.Cmp = real(Cmp);
              multi_input.Dmp = real(Dmp);
              multi_input.U = U;
              
              switch obj.Options.Ftype
                  case 'periodic'
                  %continuation
%                     for k = 1:max(size(obj.System.omega_range))
                        F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
                        F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
                        Omega = obj.System.Omega;
                        T = 2*pi/Omega;
                        h = T/obj.System.npoints;
                        tnew = 0:h:T;
                        Ft = -U.'*F_ext(tnew.').';
                        Eplus = exp(lambda_p*tnew);
                        Eminus = exp(lambda_m*tnew );

                        Gplus  = exp(lambda_p*(T-tnew))./(1-exp(lambda_p*T));
                        Gminus = exp(lambda_m*(T-tnew))./(1-exp(lambda_m*T));

                        multi_input.Eplus = Eplus;
                        multi_input.Eminus = Eminus;

                        multi_input.Gplus = Gplus;
                        multi_input.Gminus = Gminus;

                        aplus = (exp(lambda_p*h)-1)./lambda_p;
                        aminus = (exp(lambda_m*h)-1)./lambda_m;

                        multi_input.aplus = aplus;
                        multi_input.aminus = aminus;

                        bplus = (exp(lambda_p*h)-1)./(lambda_p).^2-h./lambda_p;
                        bminus = (exp(lambda_m*h)-1)./(lambda_m).^2-h./lambda_m;

                        multi_input.bplus = bplus;
                        multi_input.bminus = bminus;
                        NPer_Bm = (Gminus.*bminus-Gplus.*bplus)./(h*(lambda_m-lambda_p));
                        NPer_Am = (Gminus.*aminus-Gplus.*aplus)./(lambda_m-lambda_p) -NPer_Bm;
                        NPer_Dm = (Gminus.*lambda_m.*bminus-Gplus.*lambda_p.*bplus)./(h*(lambda_m-lambda_p));
                        NPer_Cm =  (Gminus.*aminus.*lambda_m-Gplus.*aplus.*lambda_p)./(lambda_m-lambda_p) -NPer_Dm;
                        
                        multi_input.NPer_Bm = real(NPer_Bm);
                        multi_input.NPer_Am = real(NPer_Am);
                        multi_input.NPer_Cm = real(NPer_Cm);
                        multi_input.NPer_Dm = real(NPer_Dm);

                        

                        

                        
                        
                        eta0 = 0*lambda_p;
                        etad0 = 0*lambda_p;
                        multi_input.lambda_p = lambda_p;
                        multi_input.lambda_m = lambda_m;
                        eta = zeros(2*size(M,1),max(size(tnew)));

                        for indk = 2:max(size(tnew))
                            eta0 = eta0 + multi_input.NPer_Am(:,indk).*Ft(:,indk-1) + multi_input.NPer_Bm(:,indk).*Ft(:,indk);
                            etad0 = etad0 + multi_input.NPer_Cm(:,indk).*Ft(:,indk-1) + multi_input.NPer_Dm(:,indk).*Ft(:,indk);
                            em =  (multi_input.Cm).*Ft(:,indk) + (multi_input.Dm).*Ft(:,indk-1);
                            edm = (multi_input.Cmp).*Ft(:,indk) + (multi_input.Dmp).*Ft(:,indk-1);

                            eta(1:n,indk) = (multi_input.Am).*eta(1:n,indk-1) + (multi_input.Bm).*eta(n+1:end,indk-1) + em;
                            eta(n+1:end,indk) =(multi_input.Amp).*eta(1:n,indk-1) + (multi_input.Bmp).*eta(n+1:end,indk-1) + edm;
                        end 
                        
                        DefA = (lambda_m.*Eplus -lambda_p.*Eminus)./(lambda_m-lambda_p);
                        DefB = -(Eplus -Eminus)./(lambda_m-lambda_p);
                        DefC = lambda_m.*lambda_p.*(Eplus -Eminus)./(lambda_m-lambda_p);
                        DefD = (lambda_m.*Eminus -lambda_p.*Eplus)./(lambda_m-lambda_p);

                        multi_input.DefA = real(DefA);
                        multi_input.DefB = real(DefB);
                        multi_input.DefC = real(DefC);
                        multi_input.DefD = real(DefD);
                        eta(1:n,:) = eta(1:n,:) + multi_input.DefA.*eta0 +multi_input.DefB.*etad0;
                        eta(n+1:end,:)= eta(n+1:end,:) + multi_input.DefC.*eta0 +multi_input.DefD.*etad0;

                       
                        z = [U,0*U;0*U,U]*eta;

                        A_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
                        H_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
              
                        A_dum(:,1,1,:) = z;
                        H_dum(:,1,1,:) = z;
                        A_0_1(1).coeffs = A_dum;

                        H{1}  = H_dum;
        
                        % Set up array containing index numbers at every order
                        l   = 1;
                        z_k = zeros(1,order); 
                        for k = 1:order
                            z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
                        end  

        
                        multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
                        multi_input.dt = obj.System.tforce(2);
            
                
                        multi_input.endTime = max(size(tnew));
                        %                multi_input.ttime = obj.System.tforce;
                        multi_input.ttime = tnew;
             
                        multi_input.H = H;
                        multi_input.nl_order = numel(obj.System.F);
                        multi_input.l = 1;
                        multi_input.ordering = 'revlex';
                        
%                     end
                  case 'periodic_lsq'
                  %continuation
%                     for k = 1:max(size(obj.System.omega_range))
                        F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
                        F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
                        Omega = obj.System.Omega;
                        T = 2*pi/Omega;
                        h = T/obj.System.npoints;
                        tnew = 0:h:T;
                        Ft = -U.'*F_ext(tnew.').';
                        Eplus = exp(lambda_p*tnew);
                        Eminus = exp(lambda_m*tnew );

                        Gplus  = exp(lambda_p*(T-tnew));
                        Gminus = exp(lambda_m*(T-tnew));

                        multi_input.Eplus = Eplus;
                        multi_input.Eminus = Eminus;

                        multi_input.Gplus = Gplus;
                        multi_input.Gminus = Gminus;

                        aplus = (exp(lambda_p*h)-1)./lambda_p;
                        aminus = (exp(lambda_m*h)-1)./lambda_m;

                        multi_input.aplus = aplus;
                        multi_input.aminus = aminus;

                        bplus = (exp(lambda_p*h)-1)./(lambda_p).^2-h./lambda_p;
                        bminus = (exp(lambda_m*h)-1)./(lambda_m).^2-h./lambda_m;

                        multi_input.bplus = bplus;
                        multi_input.bminus = bminus;
                        NPer_Bm = (Gminus.*bminus-Gplus.*bplus)./(h*(lambda_m-lambda_p));
                        NPer_Am = (Gminus.*aminus-Gplus.*aplus)./(lambda_m-lambda_p) -NPer_Bm;
                        NPer_Dm = (Gminus.*lambda_m.*bminus-Gplus.*lambda_p.*bplus)./(h*(lambda_m-lambda_p));
                        NPer_Cm =  (Gminus.*aminus.*lambda_m-Gplus.*aplus.*lambda_p)./(lambda_m-lambda_p) -NPer_Dm;
                        
                        multi_input.NPer_Bm = real(NPer_Bm);
                        multi_input.NPer_Am = real(NPer_Am);
                        multi_input.NPer_Cm = real(NPer_Cm);
                        multi_input.NPer_Dm = real(NPer_Dm);

                        

                        

                        
                        
                        eta0 = 0*lambda_p;
                        etad0 = 0*lambda_p;
                        multi_input.lambda_p = lambda_p;
                        multi_input.lambda_m = lambda_m;
                        eta = zeros(2*size(M,1),max(size(tnew)));

                        for indk = 2:max(size(tnew))
                            eta0 = eta0 + multi_input.NPer_Am(:,indk).*Ft(:,indk-1) + multi_input.NPer_Bm(:,indk).*Ft(:,indk);
                            etad0 = etad0 + multi_input.NPer_Cm(:,indk).*Ft(:,indk-1) + multi_input.NPer_Dm(:,indk).*Ft(:,indk);
                            em =  (multi_input.Cm).*Ft(:,indk) + (multi_input.Dm).*Ft(:,indk-1);
                            edm = (multi_input.Cmp).*Ft(:,indk) + (multi_input.Dmp).*Ft(:,indk-1);

                            eta(1:n,indk) = (multi_input.Am).*eta(1:n,indk-1) + (multi_input.Bm).*eta(n+1:end,indk-1) + em;
                            eta(n+1:end,indk) =(multi_input.Amp).*eta(1:n,indk-1) + (multi_input.Bmp).*eta(n+1:end,indk-1) + edm;
                        end 
                        
                        Period_op = cell(1,max(size(lambda_m)));
                        for indf = 1:max(size(lambda_m))
                            P = [1 ,1;lambda_p(indf), lambda_m(indf)];
                            PI = inv(P);
                            Period_op{1,indf} = real(P*[1-exp(lambda_p(indf)*T),0;0,1-exp(lambda_m(indf)*T)]*PI);
                        end    
                        P_B=blkdiag(Period_op{:});
                        multi_input.P_B = P_B;
                        Phy = zeros(2*size(M,1),1);
                        Phy(1:2:end) = eta0;
                        Phy(2:2:end) = etad0;
                        Eta_Phy =lsqminnorm(multi_input.P_B,Phy);
                        eta0 = Eta_Phy(1:2:end);
                        etad0 = Eta_Phy(2:2:end);
                        DefA = (lambda_m.*Eplus -lambda_p.*Eminus)./(lambda_m-lambda_p);
                        DefB = -(Eplus -Eminus)./(lambda_m-lambda_p);
                        DefC = lambda_m.*lambda_p.*(Eplus -Eminus)./(lambda_m-lambda_p);
                        DefD = (lambda_m.*Eminus -lambda_p.*Eplus)./(lambda_m-lambda_p);

                        multi_input.DefA = real(DefA);
                        multi_input.DefB = real(DefB);
                        multi_input.DefC = real(DefC);
                        multi_input.DefD = real(DefD);
                        eta(1:n,:) = eta(1:n,:) + multi_input.DefA.*eta0 +multi_input.DefB.*etad0;
                        eta(n+1:end,:)= eta(n+1:end,:) + multi_input.DefC.*eta0 +multi_input.DefD.*etad0;

                       
                        z = [U,0*U;0*U,U]*eta;

                        A_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
                        H_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
              
                        A_dum(:,1,1,:) = z;
                        H_dum(:,1,1,:) = z;
                        A_0_1(1).coeffs = A_dum;

                        H{1}  = H_dum;
        
                        % Set up array containing index numbers at every order
                        l   = 1;
                        z_k = zeros(1,order); 
                        for k = 1:order
                            z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
                        end  

        
                        multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
                        multi_input.dt = obj.System.tforce(2);
            
                
                        multi_input.endTime = max(size(tnew));
                        %                multi_input.ttime = obj.System.tforce;
                        multi_input.ttime = tnew;
             
                        multi_input.H = H;
                        multi_input.nl_order = numel(obj.System.F);
                        multi_input.l = 1;
                        multi_input.ordering = 'revlex';
                        
%                     end
                  case 'aperiodic'
                      F_ext1 = griddedInterpolant(obj.System.tforce(:),obj.System.Fext.','linear');
                      F_ext= @(xa) ((xa >=obj.System.tforce(1)) & (xa <=obj.System.tforce(end))).*F_ext1(xa);
              
                      h = obj.System.dh; 
                      tnew = 0:h:obj.System.tforce(end);
              
                      Ft = -U.'*F_ext(tnew.').';
             
                       eta = zeros(2*size(U,2),max(size(tnew)));
                       np = size(U,2);
                        for indk = 2:max(size(tnew))
                            eta(1:np,indk) = (multi_input.Am).*eta(1:np,indk-1) + (multi_input.Bm).*eta(np+1:end,indk-1) + (multi_input.Cm).*Ft(:,indk) + (multi_input.Dm).*Ft(:,indk-1);
                            eta(np+1:end,indk) =(multi_input.Amp).*eta(1:np,indk-1) + (multi_input.Bmp).*eta(np+1:end,indk-1) + (multi_input.Cmp).*Ft(:,indk) + (multi_input.Dmp).*Ft(:,indk-1);
                        end 
                        z = [U,0*U;0*U,U]*eta;

                        A_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
                        H_dum = zeros(2*(obj.System.n),1,1,max(size(tnew)));
              
                        A_dum(:,1,1,:) = z;
                        H_dum(:,1,1,:) = z;
                        A_0_1(1).coeffs = A_dum;

                        H{1}  = H_dum;
        
                        % Set up array containing index numbers at every order
                        l   = 1;
                        z_k = zeros(1,order); 
                        for k = 1:order
                            z_k(k) = nchoosek(k+l-1,l-1); % At order k, SSM dim l there are z_k(k,l) spatial multi-indices
                        end  

        
                        multi_input.Z_cci = z_k; %in this case there is no conjugate center index, the full index set is treated
                        multi_input.dt = obj.System.tforce(2);
            
                
                        multi_input.endTime = max(size(tnew));
                        %                multi_input.ttime = obj.System.tforce;
                        multi_input.ttime = tnew;
             
                        multi_input.H = H;
                        multi_input.nl_order = numel(obj.System.F);
                        multi_input.l = 1;
                        multi_input.ordering = 'revlex';

              end    
                 
              
              
                
              
  
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