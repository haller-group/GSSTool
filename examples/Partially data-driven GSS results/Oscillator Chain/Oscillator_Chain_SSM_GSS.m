%% Oscillator chain under stochastic forcing — SSM+GSS vs. AE+LSTM prediction
% Learns a 2D data-driven SSM (SSMLearn-style, via fastSSM) from free
% trajectories of a stochastically-forced oscillator chain, adds a GSS
% correction to account for the forcing, and compares the resulting
% prediction against a full-order time integration and an
% autoencoder+LSTM surrogate. Also renders a comparison movie
% (ssm_oc_fast.mp4).

clear all; close all; clc
% run ../../install.m
exampleDir = fileparts(mfilename('fullpath'));
cd(exampleDir)
%% Example setup

n = 20;
m = 0.1;
k = 100;
c = 0.1;

kappa2 = 0;
kappa3 = 2500;

[M,C,K,fnl,~] = build_model(n,m,c,k,kappa2,kappa3);
%% Dynamical system setup 

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',5,'Nmax',20,'notation','multiindex')
DS.order = 2;

[V, lambda, W] = eig(full(inv(DS.B)*DS.A));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) ); % mass normalized VMs
DS.W = (W*diag(1./(sqrt(mu)')));%O.V.';
DS.lambda = diag(lambda); % is now a vector

%%  Noise generation


f_0_1 = zeros(n,1);
f_0_1(1) = 1; 
 
f_0_2 = zeros(n,1);
f_0_2(end) = 1; 

variance = 3.4;
Signal_size = 30001;
tspan = linspace(0,100,Signal_size);
Noise_model_1 = sqrt(variance).*randn(1,Signal_size);
Noise_model_2 = sqrt(variance).*randn(1,Signal_size);
Fs = 100; % Sampling rate in Hz
T = 100; % Duration in seconds
f_cutoff = 40;
order = 6; % Filter order
[B1, A1] = butter(order, f_cutoff/(Fs/2), 'low');
filteredSignal_1 = filter(B1, A1, Noise_model_1);
filteredSignal_2 = filter(B1, A1, Noise_model_2);
figure 
plot(tspan, filteredSignal_1,'color','red')
hold on 
plot(tspan, filteredSignal_2,'color','blue')

%% 
% Load AE+LSTM predictions. Test3 denotes unseen forcing.

Test3 = load('0.50_oscillatorchain_nElements_20_latdim_9_test3.mat');
F1 = Test3.force_orig;
LSTM_x = Test3.x_processed;
LSTM_t = Test3.t;

Fext_T = f_0_1*(-F1(:,1).') + f_0_2*(-F1(:,2).');

%% Dynamical system forcing setup
h = 0.01;
Fext = Fext_T;
nc_order = 1;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h); 
DS.ncc_order = nc_order;

tspany = linspace(0,100,30001);
F_ext1 = griddedInterpolant(DS.tforce(:),DS.Fext.','linear');
F_ext_full= @(xa) (((xa >=DS.tforce(1)) & (xa <=DS.tforce(end))).*(F_ext1(xa)).');

Fext = F_ext_full(tspany);
h = tspany(2);
set(DS,'Fext',Fext,'tforce',tspany,'tend',tspany(end)-tspany(1),'dh',h); 
DS.ncc_order = nc_order;


%% Compute GSS


S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')

order = 10;
ticstart = tic;
[A_0_P,timerPE] = S.compute_anchor(order);
time_PE = toc(ticstart);

%%
%Output anchor
epsilon = 1;
order =10;

[A_0_outP] = output_anchor(A_0_P,order,epsilon);


order =1;

[A_0_outPL] = output_anchor(A_0_P,order,epsilon);

%%
epsilon = 1;
F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext= @(xa) epsilon*(((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa)).';
             
%% Learn 2D SSM from data.  

startt = tic;

ICRadius = 1;
nTraj = 4;

[F, lambda, V, G, DG,G_nl,A] = functionFromTensors(M, C, K, fnl);

tic
[V,D] = eig(full(A));
            dsa = diag(D);
            [d,indy]=sort(real(dsa));
            Ds=D(indy,indy);         %sorted eigenvalue matrix
            Vs=V(:,indy); 
 Vnew = [real(Vs(:,1:2:end)),imag(Vs(:,1:2:end))];
 Vnew_main = [];
 for indt = 1:20
        Vnew_main = [Vnew_main,Vnew(:,indt),Vnew(:,indt+20)];
 end  
 INVnew = inv(Vnew_main); 
                VT = INVnew(end-1:end,:).';
IC = [10*Vnew_main(:,40),10*Vnew_main(:,39),10*Vnew_main(:,38),10*Vnew_main(:,37)];
store ={};
for index = 1:nTraj
q0 = IC(1:n,index);
qd0 = IC(n+1:end,index);
tmax = (tspan(end)-tspan(1))*2;
qdd0 = -inv(M)*(K*q0+C*qd0+G_nl(IC(:,index)));
TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);

residual = @(q,qd,qdd,t) residual_nonlinear_noforce( q, qd, qdd, t, M,K,C,G_nl);

TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);
store{index} = TI_NL;
end
toc
%%
% plot the actual data 

figure
traj_train =[store{1,1}.Solution.q;store{1,1}.Solution.qd];
traj_test =[store{1,2}.Solution.q;store{1,2}.Solution.qd];
indx = 10;
indy = indx+20;
indz = 11;


tr = 3000;
plot3(traj_train(indx,tr:end),traj_train(indy,tr:end),traj_train(indz,tr:end),'-','Linewidth',3,'color','red')
hold on 
plot3(traj_test(indx,tr:end),traj_test(indy,tr:end),traj_test(indz,tr:end),'-','Linewidth',3,'color','blue')
grid on
legend('Training trajectory','Testing trajectory')

%%
% Construct data matrix for fastSSM input
tic
yData = cell(1,2); 
for ind = 1:1
    tr = 3000;
    time = store{1,ind}.Solution.time(tr:end)-store{1,ind}.Solution.time(tr);
    data_y = [store{1,ind}.Solution.q(:,tr:end);store{1,ind}.Solution.qd(:,tr:end)]; 
    yData{ind,1} = time;
    yData{ind,2} = data_y;
end    
[Mmap, iMmap, Tmap, iTmap, Nflow, yRec,fnl_rom,RMap,R,Mr] = fastSSM(yData, 11, VT,1);
toc
%%
yDataTest = cell(1,2); 
for ind = 2:2
    tr = 3000;
    time = store{1,ind}.Solution.time(tr:end)-store{1,ind}.Solution.time(tr);
    data_y = [store{1,ind}.Solution.q(:,tr:end);store{1,ind}.Solution.qd(:,tr:end)]; 
    yDataTest{1,1} = time;
    yDataTest{1,2} = data_y;
end  
Xi=iMmap(yDataTest{1,2}(:,1));
[~, zRec] = ode45(RMap, yDataTest{1,1}, Xi, odeset('RelTol', 1e-6));
yRec = Mmap((zRec.'));
%%
%%
% plot the actual data 

figure
traj_train =[store{1,1}.Solution.q;store{1,1}.Solution.qd];
traj_test =[store{1,2}.Solution.q;store{1,2}.Solution.qd];
indx = 10;
indy = indx+20;
indz = 11;
refine_c = 20;
[Xi1,Xi2] = meshgrid(linspace(-5.4,5.4,refine_c));
Xi1_c = Xi1(:);
Xi2_c = Xi2(:);
Xi_c = [Xi1_c,Xi2_c].';
SSM_map = Mmap((Xi_c));
SSM_x= reshape(SSM_map(indx,:),refine_c,refine_c);
SSM_y= reshape(SSM_map(indy,:),refine_c,refine_c);
SSM_z= reshape(SSM_map(indz,:),refine_c,refine_c);

tr = 8000;
h =surf(SSM_x,SSM_y,SSM_z);
 h.EdgeColor = 'black';
       h.EdgeAlpha = 0.5;
        h.FaceColor = 'cyan';
       h.FaceAlpha = 0.3;
      h.FaceLighting = 'gouraud';
      axis tight ; % Keep axis limits stable
       axis vis3d ; % Fix the aspect ratio for 3D rotation

hold on
plot3(traj_train(indx,tr:end),traj_train(indy,tr:end),traj_train(indz,tr:end),'-','Linewidth',3,'color',[0 0 0])
hold on 
plot3(traj_test(indx,tr:end),traj_test(indy,tr:end),traj_test(indz,tr:end),'-','Linewidth',3,'color',[0.7 0.7 0.7 0.5])
hold on

grid on
view(-61,22)
l1 = legend('$O(11)$ SSM (2D)','Training trajectory','Testing trajectory','Interpreter','latex');
l1.Position = [0.1985 0.6717 0.2077 0.0890];
xlabel('$x_{10}$ $[\mathrm{m}]$','Interpreter','latex')
ylabel('$\dot{x}_{10}$ $[\mathrm{m}/s]$','Interpreter','latex')
zlabel('$x_{11}$ $[\mathrm{m}]$','Interpreter','latex')

%%
% figure
% hold on 

figure
plot(yDataTest{1,1}, yDataTest{1,2}(1,:),'-','Linewidth',3,'color','black')
hold on
plot(yDataTest{1,1},yRec(1,:),'--','Linewidth',3,'color','red')
hold on 
xlim([0 30])
legend('True','$O(11)$ SSM ($\mathrm{2D}$) prediction','Interpreter','latex')
xlabel('$t$ $[\mathrm{s}]$','Interpreter','latex')
ylabel('$x_1$ $[\mathrm{m}]$','Interpreter','latex')


%%
% Generally forced SSM-reduced model

Forced_map = @(t,xi) RMap(t,xi) - iMmap([0*F_ext(t);inv(M)*F_ext(t)]);
Xi=iMmap(yDataTest{1,2}(:,1))*0;
[~, zRec] = ode45(Forced_map, tspany, Xi, odeset('RelTol', 1e-6));
yRec = Mmap((zRec.'));

%%
% O(\Delta) correction to the SSM 


lambda = S.System.lambda;
V = S.System.V;
W = S.System.W;

[V,D] = eig(full(A));
dsa = diag(D);
lambda = dsa;
W = inv(V)';
[d,indy]=sort(-real(dsa));
Ds=D(indy,indy);         %sorted eigenvalue matrix
Vs=V(:,indy); 
lambda = diag(Ds);
modal_trunc = 40;
lambda = lambda(1:modal_trunc);
W = inv(Vs);
W = W(1:modal_trunc,:)';
V = Vs(:,1:modal_trunc);
F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext= @(xa) ((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa);
h = tspany(2);

tnew = 0:h:S.System.tforce(end);
              
              Ft  = [sparse(S.System.n,max(size(tnew)));-inv(M)*F_ext(tnew.').'];
          tic
              phi = W' *(eye(2*n)-Vnew_main(:,end-1:end)*VT')* Ft;
              Ftn = phi(:,1:end-1);
              Ftn_p1 = phi(:,2:end);
              ExpoA = (exp(lambda*h)-1)./(lambda);
              ExpoB = (exp(lambda*h))./(lambda.^2) - 1./(lambda.^2) - h./(lambda);
              PHI = ExpoA.*Ftn + ExpoB.*(Ftn_p1-Ftn)/h;
              et = 0:h:(S.System.tforce(end)-S.System.tforce(1));
              S.System.Explambda = exp(lambda.*et(1:end-1));
              
              multi_input.ExpoA = ExpoA;
              multi_input.ExpoB = ExpoB;
              multi_input.h = h;
               eta = zeros(modal_trunc,max(size(tnew)));
              for j = 1:modal_trunc %1:2 %1:2*size(M,1)  
                    eta_d = conv(PHI(j,:),S.System.Explambda(j,:));
                    ETA_M = eta_d(1:max(size(et(1:end-1))));
                    eta(j,:) = [0,ETA_M];
              end
           
                
               z = real(V*eta);
toc 
               yRec = yRec + epsilon*z;
              
%%
figure 
plot(zRec(:,1),zRec(:,2))
%

F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext= @(xa) epsilon*(((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa)).';

%
startt = tic;

IC = yRec(:,1);
q0 = IC(1:n,1);
qd0 = IC(n+1:end,1);
tmax = (tspany(end)-tspany(1));
qdd0 = -M\(F_ext(0)) -K\q0;
[F, lambda, V, G, DG,G_nl] = functionFromTensors(M, C, K, fnl);
TI_NL_force = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);

residual = @(q,qd,qdd,t) residual_nonlinear_slow( q, qd, qdd, t, M,K,C,G_nl, F_ext);

TI_NL_force.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);

%%
figure 
plot(tspany,yRec(1,:),'-','Linewidth',3,'color','red')
hold on 
plot(TI_NL_force.Solution.time,TI_NL_force.Solution.q(1,:),'-','Linewidth',3,'color',[0 0 0 0.5])
legend('SSMLearn','True')
xlabel('$t$','Interpreter','latex')
ylabel('$x_1$','Interpreter','latex')
%% GSS prediction on SSM-reduced model 
fnlr = fnl_rom(2:end);
Lin = double(fnl_rom{1});
DS = DynamicalSystem();
Fcheck  =set_fnl_fake(fnlr);
set(DS,'A',Lin,'B',eye(2),'F',Fcheck);
set(DS.Options,'Emax',5,'Nmax',20,'notation','multiindex')
DS.order = 1;

[V, lambda, W] = eig(full(inv(DS.B)*DS.A));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) ); % mass normalized VMs
DS.W = (W*diag(1./(sqrt(mu)')));%O.V.';
DS.lambda = diag(lambda); % is now a vector

F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext= @(xa) (((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa)).';

Fext = iMmap([0*F_ext(tspany.');inv(M)*F_ext(tspany.')]);
h = tspany(2);
set(DS,'Fext',Fext,'tforce',tspany,'tend',tspany(end)-tspany(1),'dh',h); 
DS.ncc_order = nc_order;

Gs = SSM(DS);
set(Gs.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Piecewise_exact_first_order')
time_PE = 0;

order = 5;
ticstart = tic;
[A_0_P,timerPE] = Gs.compute_anchor(order);
time_PE =time_PE+ toc(ticstart);


order =5;

[A_0_outP] = output_anchor(A_0_P,order,epsilon);

order =1;

[A_0_outPL] = output_anchor(A_0_P,order,epsilon);


figure 
plot(zRec(:,1),zRec(:,2),'Linewidth',3,'color',[0 0 0 0.5])
hold on 
plot(A_0_outP(1,:),A_0_outP(2,:),'Linewidth',3,'color',[1 0 0 0.5])
hold on 
plot(A_0_outPL(1,:),A_0_outPL(2,:),'Linewidth',3,'color',[0 1 0 0.5])
legend('ROM advection with 0 initial condition','GSS O(10)','GSS O(1)')

figure
inde = 1;
plot(tspany,zRec(:,inde),'Linewidth',3,'color',[0 0 0 0.5])
hold on 
plot(tspany,A_0_outP(inde,:),'Linewidth',3,'color',[1 0 0 0.5])
hold on
plot(tspany,A_0_outPL(inde,:),'Linewidth',3,'color',[0 1 0 0.5])

%%
yRec_anchor = Mmap((A_0_outP))+ epsilon*z;
yRec_anchor_linear = Mmap((A_0_outPL))+ epsilon*z;


figure 
indexf = 1;
indexfd = 1;
plot(TI_NL_force.Solution.time,TI_NL_force.Solution.q(indexfd,:)/(max(abs(TI_NL_force.Solution.q(indexfd,:)))),'-','Linewidth',3,'color',[0 0 0])
hold on 
plot(tspany,yRec_anchor(indexf,:)/(max(abs(yRec_anchor(indexf,:)))),'--','Linewidth',3,'color',[1 0 0])
hold on 
% hold on 
plot(LSTM_t,LSTM_x(:,indexf),'-','LineWidth',3,'color',[0 1 0 0.5])
hold on
xlim([80 85])
legend('True','GSS $O(5)$ from $O(11)$ SSM $(\mathrm{2D})$','Autoencoder + LSTM','Interpreter','latex')%,'GSS O(1)')

xlabel('$t$ $[\mathrm{s}]$','Interpreter','latex')
ylabel('$x_1$ $[\mathrm{m}]$','Interpreter','latex')


%%
zer = iMmap([TI_NL_force.Solution.q;TI_NL_force.Solution.qd]);
zer2 = iMmap(yRec);

figure 
plot(zRec(:,1),zRec(:,2),'-','Linewidth',3,'color','red')
hold on 
plot(zer(1,:),zer(2,:),'--','Linewidth',3,'color','blue')
% hold on 

legend('SSMLearn','True')
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')

figure 
plot(tspany,zRec(:,2),'-','Linewidth',3,'color','red')
hold on 
% hold on
plot(TI_NL_force.Solution.time,zer(2,:),'--','Linewidth',3,'color','blue')
legend('SSMLearn','True')
xlabel('$t$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')

figure 
index =20;
plot3(zer(1,:),zer(2,:),TI_NL_force.Solution.q(index,:),'-','Linewidth',3,'color',[0 0 0 0.5])
hold on 
plot3(A_0_outP(1,:),A_0_outP(2,:),yRec_anchor(index,:),'-','Linewidth',3,'color',[1 0 0 0.5])
hold on
plot3(A_0_outPL(1,:),A_0_outPL(2,:),yRec_anchor_linear(index,:),'-','Linewidth',3,'color',[0 1 0 0.5])

% hold on 

legend('True','GSS $O(5)$','GSS $O(1)$','Interpreter','latex')%,'GSS O(1)')

xlabel('$\eta_1$','Interpreter','latex')
ylabel('$\eta_2$','Interpreter','latex')
zlabel('$x_1$','Interpreter','latex')
grid on 
box on 
%%
% plot of the SSM in the physical space with AE+LSTM results.


LSTM_x_unnormalized = LSTM_x.*max(abs(TI_NL_force.Solution.q).');

[LSTM_v, LSTM_x_trunc,ind_LSTM] = ftd_full(LSTM_x_unnormalized.', LSTM_t);
LSTM_t_trunc = LSTM_t(ind_LSTM);

%%
figure 
index =20;
indx = 10;
indy = indx+20;
indz = 11;
LSTM_full = [LSTM_x_trunc;LSTM_v];
Full_traj = [TI_NL_force.Solution.q;TI_NL_force.Solution.qd];
Full_traj = Full_traj(:,ind_LSTM);
yRec_anchor_chop = yRec_anchor(:,ind_LSTM);
ti = 3000;
len = 100;
z_trunc = z(:,ind_LSTM);
refine_c = 20;
[Xi1,Xi2] = meshgrid(linspace(-2.4,2.4,refine_c));
Xi1_c = Xi1(:);
Xi2_c = Xi2(:);
Xi_c = [Xi1_c,Xi2_c].';
SSM_map = Mmap((Xi_c))+ epsilon*z_trunc(:,ti+len);

SSM_x= reshape(SSM_map(indx,:),refine_c,refine_c);
SSM_y= reshape(SSM_map(indy,:),refine_c,refine_c);
SSM_z= reshape(SSM_map(indz,:),refine_c,refine_c);

h =surf(SSM_x,SSM_y,SSM_z);
 h.EdgeColor = 'black';
       h.EdgeAlpha = 0.5;
        h.FaceColor = 'cyan';
       h.FaceAlpha = 0.3;
      h.FaceLighting = 'gouraud';
      axis tight ; % Keep axis limits stable
       axis vis3d ; % Fix the aspect ratio for 3D rotation
hold on
plot3(Full_traj(indx,ti:ti+len),Full_traj(indy,ti:ti+len),Full_traj(indz,ti:ti+len),'-','Linewidth',3,'color',[0 0 0 0.5])
hold on 
plot3(yRec_anchor_chop(indx,ti:ti+len),yRec_anchor_chop(indy,ti:ti+len),yRec_anchor_chop(indz,ti:ti+len),'-','Linewidth',3,'color',[1 0 0 0.5])
hold on
plot3(LSTM_full(indx,ti:ti+len),LSTM_full(indy,ti:ti+len),LSTM_full(indz,ti:ti+len),'-','Linewidth',3,'color',[0 1 0 0.5])
hold on

plot3(Full_traj(indx,ti+len),Full_traj(indy,ti+len),Full_traj(indz,ti+len),'.','MarkerSize',25,'color',[0 0 0])
hold on 
plot3(yRec_anchor_chop(indx,ti+len),yRec_anchor_chop(indy,ti+len),yRec_anchor_chop(indz,ti+len),'.','MarkerSize',25,'color',[1 0 0])
hold on
plot3(LSTM_full(indx,ti+len),LSTM_full(indy,ti+len),LSTM_full(indz,ti+len),'.','MarkerSize',25,'color',[0 1 0])
hold on

grid on
box on
view(-61,22)
l3 = legend('O(11) SSM $(\mathrm{2D})$','True','GSS $O(5)$ from $O(11)$ SSM $(\mathrm{2D})$','Autoencoder + LSTM','Interpreter','latex')%,'GSS O(1)')

l3.Position = [0.1985 0.6717 0.2077 0.0890];
xlabel('$x_{10}$ $[\mathrm{m}]$','Interpreter','latex')
ylabel('$\dot{x}_{10}$ $[\mathrm{m}/s]$','Interpreter','latex')
zlabel('$x_{11}$ $[\mathrm{m}]$','Interpreter','latex')
time_quote = strcat("$t =$", num2str(round(LSTM_t_trunc(ti+len),2)));
title(time_quote,'Interpreter','latex')

%%
hFig = figure;
 

index =20;
indx = 1;
indy = indx+20;
indz = 20;
LSTM_full = [LSTM_x_trunc;LSTM_v];
Full_traj = [TI_NL_force.Solution.q;TI_NL_force.Solution.qd];
Full_traj = Full_traj(:,ind_LSTM);
yRec_anchor_chop = yRec_anchor(:,ind_LSTM);
ti = 24001;
len = 100;
z_trunc = z(:,ind_LSTM);
clear movieVector

for ind=1:25501-ti+1
clf
ti = 24001+ind-1;
refine_c = 20;
[Xi1,Xi2] = meshgrid(linspace(-2.4,2.4,refine_c));
Xi1_c = Xi1(:);
Xi2_c = Xi2(:);
Xi_c = [Xi1_c,Xi2_c].';
SSM_map = Mmap((Xi_c))+ epsilon*z_trunc(:,ti+len);

SSM_x= reshape(SSM_map(indx,:),refine_c,refine_c);
SSM_y= reshape(SSM_map(indy,:),refine_c,refine_c);
SSM_z= reshape(SSM_map(indz,:),refine_c,refine_c);

h =surf(SSM_x,SSM_y,SSM_z);
 h.EdgeColor = 'black';
       h.EdgeAlpha = 0.5;
        h.FaceColor = 'cyan';
       h.FaceAlpha = 0.3;
      h.FaceLighting = 'gouraud';
      axis tight ; % Keep axis limits stable
       axis vis3d ; % Fix the aspect ratio for 3D rotation
hold on
plot3(LSTM_full(indx,ti:ti+len),LSTM_full(indy,ti:ti+len),LSTM_full(indz,ti:ti+len),'-','Linewidth',3,'color',[0 1 0 0.5])
hold on
plot3(Full_traj(indx,ti:ti+len),Full_traj(indy,ti:ti+len),Full_traj(indz,ti:ti+len),'-','Linewidth',3,'color',[0 0 0 0.5])
hold on 
plot3(yRec_anchor_chop(indx,ti:ti+len),yRec_anchor_chop(indy,ti:ti+len),yRec_anchor_chop(indz,ti:ti+len),'-','Linewidth',3,'color',[1 0 0 0.5])
hold on

plot3(LSTM_full(indx,ti+len),LSTM_full(indy,ti+len),LSTM_full(indz,ti+len),'.','MarkerSize',25,'color',[0 1 0])
hold on
plot3(Full_traj(indx,ti+len),Full_traj(indy,ti+len),Full_traj(indz,ti+len),'.','MarkerSize',25,'color',[0 0 0])
hold on 
plot3(yRec_anchor_chop(indx,ti+len),yRec_anchor_chop(indy,ti+len),yRec_anchor_chop(indz,ti+len),'.','MarkerSize',25,'color',[1 0 0])
hold on

grid on
box on
view(-61,21)
 axis([-0.03,0.03,-1.2,1.2,-0.03,0.03])
xl = xlabel('$x_{1}$ $[\mathrm{m}]$','Interpreter','latex');
 xl.Position = [-0.0034 -1.0491 -0.0128];
yl = ylabel('$\dot{x}_{1}$ $[\mathrm{m}/s]$','Interpreter','latex');
 yl.Position = [-0.0447 0.1026 -0.0467];
zl = zlabel('$x_{20}$ $[\mathrm{m}]$','Interpreter','latex');
time_quote = strcat("$t =$", num2str(round(LSTM_t_trunc(ti+len),2)));
title(time_quote,'Interpreter','latex')

set(findall(hFig, '-property', 'FontSize'), 'FontSize', 18); % Set to 18 as desired

grid on;
box on
set(gca, 'FontSize', 18);
set(gca, 'TickLabelInterpreter' , 'latex');

set(gcf,'color','white')

set(gcf,'color','white')
set(gcf,'Renderer','painters')


set(hFig, 'Units' , 'Inches' );
pos = get(hFig, 'Position' );
set(hFig, 'PaperPositionMode' , 'Auto' , 'PaperUnits' , 'Inches' , 'PaperSize' ,[pos(3), pos(4)])

 movieVector(ind) = getframe(hFig);
end
%%
myWriter = VideoWriter('ssm_oc_fast','MPEG-4');
myWriter.FrameRate = 50;

open(myWriter);
% 
for indj = 1:max(size(movieVector))
writeVideo(myWriter,movieVector(indj));
end

close(myWriter);