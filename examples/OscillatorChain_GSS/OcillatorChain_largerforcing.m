%% Oscillator chain — larger-forcing convergence study (O(1)/O(5)/O(10)/O(15))
% Oscillator chain subject to unbounded filtered Gaussian noise, forced by
% a recorded/data-driven signal (Test3 dataset, unseen forcing). Compares
% the full-order time integration against the aSSM anchor trajectory at
% increasing truncation orders, to check convergence at this larger
% forcing level (companion to Oscillator_chain_LTSM_compare_GSS.m).
%
% Requires 0.50_oscillatorchain_nElements_20_latdim_9_test3.mat in the
% working directory (same as Oscillator_chain_LTSM_compare_GSS.m and
% rw_calculation_OscillatorChain.m).

clear all; close all; clc
run ../../install.m

%% Example setup
n = 20;
m = 0.1;
k = 100;
c = 0.1;
kappa2 = 0;
kappa3 = 2500;

[M,C,K,fnl,~] = build_model(n,m,c,k,kappa2,kappa3);

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',5,'Nmax',20,'notation','multiindex')
DS.order = 2;

%% Noise generation (diagnostic signal plot only; forcing itself comes from Test3 below)
f_0_1 = zeros(n,1);
f_0_1(1) = 1;
f_0_2 = zeros(n,1);
f_0_2(end) = 1;

variance = 3.4;
Signal_size = 10001;
tspan = linspace(0,100,Signal_size);
Noise_model_1 = sqrt(variance).*randn(1,Signal_size);
Noise_model_2 = sqrt(variance).*randn(1,Signal_size);
Fs = 100; % Sampling rate in Hz
f_cutoff = 40;

order = 6; % Filter order
[B1, A1] = butter(order, f_cutoff/(Fs/2), 'low');
filteredSignal_1 = filter(B1, A1, Noise_model_1);
filteredSignal_2 = filter(B1, A1, Noise_model_2);

figure('Position',[100, 100, 700, 300])
plot(tspan, filteredSignal_1,'color','red')
hold on
plot(tspan, filteredSignal_2,'color','blue')
xlabel('$t$ [s]','Interpreter','latex');
set(gcf,'color','white')

%% Load forcing/comparison data (Test3 = unseen forcing used for validation)
Test3 = load('0.50_oscillatorchain_nElements_20_latdim_9_test3.mat');
F1 = Test3.force_orig;
LSTM_x = Test3.x_processed;
LSTM_t = Test3.t;

Fext_T = f_0_1*(-F1(:,1).') + f_0_2*(-F1(:,2).');

%% Dynamical system forcing setup
tspan = Test3.t;
h = 0.01;
Fext = Fext_T;
nc_order = 1;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
DS.ncc_order = nc_order;

tspany = linspace(0,100,30001);
F_ext1 = griddedInterpolant(DS.tforce(:),DS.Fext.','linear');
F_ext_full = @(xa) (((xa >=DS.tforce(1)) & (xa <=DS.tforce(end))).*(F_ext1(xa)).');

Fext = F_ext_full(tspany);
h = tspany(2);
set(DS,'Fext',Fext,'tforce',tspany,'tend',tspany(end)-tspany(1),'dh',h);
DS.ncc_order = nc_order;

%% Compute the aSSM anchor trajectory
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')
order = 15;
ticstart = tic;
[A_0_P,timerPE] = S.compute_anchor(order);
time_PE = toc(ticstart);

%% Output anchor at increasing truncation orders
epsilon = 1.8;

order = 5;
[A_0_outP5] = output_anchor(A_0_P,order,epsilon);
order = 10;
[A_0_outP10] = output_anchor(A_0_P,order,epsilon);
order = 15;
[A_0_outP15] = output_anchor(A_0_P,order,epsilon);
order = 1;
[A_0_outPL] = output_anchor(A_0_P,order,epsilon);

F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext = @(xa) epsilon*(((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa)).';

%% Full-order time integration, for comparison against the anchor
startt = tic;
h = tspany(2);
q0  = A_0_outP15(1:n,1)*0;
qd0 = A_0_outP15(n+1:end,1)*0;
tmax = tspan(end)-tspan(1);
qdd0 = -M\(F_ext(0)) - K\q0;
[F, lambda, V, G, DG, G_nl] = functionFromTensors(M, C, K, fnl);

TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);
residual = @(q,qd,qdd,t) residual_nonlinear_slow( q, qd, qdd, t, M,K,C,G_nl, F_ext);
TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);

%% Plot: full-order trajectory vs. AE+LSTM prediction vs. aSSM anchor at O(1)/O(5)/O(10)/O(15)
figure
indexR = 10;
t_sim = TI_NL.Solution.time;

plot(t_sim(1:2:end),TI_NL.Solution.q(indexR,1:2:end)/(max(abs(TI_NL.Solution.q(indexR,1:2:end)))),'-','LineWidth',3,'color','black')
hold on
plot(LSTM_t,LSTM_x(:,indexR),'-','LineWidth',3,'color',[0 1 0 0.5])
hold on
plot(tspany(1:2:end),A_0_outPL(indexR,1:2:end)/(max(abs(A_0_outPL(indexR,1:2:end)))),'-','LineWidth',3,'color',[0 0 1 0.5])
hold on
plot(tspany(1:2:end),A_0_outP5(indexR,1:2:end)/(max(abs(A_0_outP5(indexR,1:2:end)))),'-','LineWidth',3,'color',[1.00 0.41 0.16 0.5])
hold on
plot(tspany(1:2:end),A_0_outP10(indexR,1:2:end)/(max(abs(A_0_outP10(indexR,1:2:end)))),'-','LineWidth',3,'color',[1 0 0 0.5])
hold on
plot(tspany(1:2:end),A_0_outP15(indexR,1:2:end)/(max(abs(A_0_outP15(indexR,1:2:end)))),'-','LineWidth',3,'color',[0.65 0.16 0.16 0.5])

lgd = legend('Full model','AE+LSTM prediction','O(1) GSS','O(5) GSS','O(10) GSS','O(15) GSS', ...
    'Interpreter','latex','Location','northoutside');
lgd.NumColumns = 3;

title('$\Delta = 12.82$ [N]','Interpreter','latex')
string = strcat('$\frac{x_{',num2str(10),'}(t)}{\|x_{10}(t)\|}$');
xlabel('$t$ [s]','Interpreter','latex');
ylabel(string,'Interpreter','latex');
xlim([74 78])
ylim([-1 1])
grid on
