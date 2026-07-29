%% Oscillator chain — periodic forcing: forced-response sweep
% Computes a forced-response curve for an oscillator chain with cubic and
% quadratic nonlinearities under periodic forcing, sweeping the forcing
% frequency Omega. At each frequency, the maximum response amplitude is
% compared across four models:
%   - "True": full-order (Newmark) time integration
%   - the exact linear solution
%   - the aSSM anchor under periodic forcing (Ftype 'periodic_lsq'), read
%     over a single period
%   - the aSSM anchor under aperiodic forcing (Ftype 'aperiodic'),
%     integrated over many periods and read off at steady state
%
% Forced system:
%   M*xddot + C*xdot + K*x + f(x,xdot) = epsilon*f_ext(Omega*t)
% with periodic forcing f_ext(phi) = f_0*cos(phi) = (f_0/2)*e^{i phi} + (f_0/2)*e^{-i phi}.

clear all; close all; clc
run ../../install.m
exampleDir = fileparts(mfilename('fullpath'));
cd(exampleDir)

%% Example setup
n = 10;
m = 1;
k = 1;
c = 3;
kappa2 = 0.5;
kappa3 = 1;

[M,C,K,fnl,~] = build_model(n,m,c,k,kappa2,kappa3);

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',5,'Nmax',10,'notation','multiindex')

%% Periodic forcing: Fourier coefficients
epsilon = 50e-3;
f_0 = ones(n,1);

kappas = [-1; 1];
coeffs = [f_0 f_0]/2;
DS.add_forcing(coeffs, kappas, epsilon);
fext = f_0;

%% Linear modal analysis, master subspace, and forcing-frequency sweep range
[V,D,W] = DS.linear_spectral_analysis();

S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex')
masterModes = [1,2];
S.choose_E(masterModes);

omega0 = imag(S.E.spectrum(1));
omegaRange = omega0*[0.5 1.5];
omegaRange = linspace(omegaRange(1),omegaRange(2),50);

%% Sweep the forcing frequency and record the max response amplitude
index = floor(n/2);
MaxAmp_Per    = [];
MaxAmp_Aper   = [];
MaxAmp_Linear = [];
MaxAmp_true   = [];

time_PE_I = 0;
time_PE   = 0;

for indf = 1:max(size(omegaRange))

    %% aSSM anchor under periodic forcing (single period)
    DS.Omega = omegaRange(indf);
    DS.npoints = 100;
    Fext_f = @(t,a) fext*cos(a*t);
    T = 2*pi/DS.Omega;
    h = T/DS.npoints;
    N = 1;
    tspan = 0:h:N*T;
    Fext = Fext_f(tspan,DS.Omega);
    nc_order = 1;

    set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
    DS.ncc_order = nc_order;

    S = SSM(DS);
    set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','periodic_lsq')
    order = 10;
    ticstart = tic;
    [A_0_P_I,timerPE_I] = S.compute_anchor(order);
    time_PE_I = time_PE_I + toc(ticstart);

    order = 10;
    [A_0_outP_I] = output_anchor(A_0_P_I,order,epsilon);
    MaxAmp_Per = [MaxAmp_Per, max(abs(A_0_outP_I(index,:)))];

    %% aSSM anchor under aperiodic forcing, integrated over many periods to steady state
    DS.Omega = omegaRange(indf);
    DS.npoints = 100;
    Fext_f = @(t,a) fext*cos(a*t);
    T = 2*pi/DS.Omega;
    h = T/DS.npoints;
    N = 20;
    tspan = 0:h:N*T;
    Fext = Fext_f(tspan,DS.Omega);
    nc_order = 1;

    set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
    DS.ncc_order = nc_order;

    S = SSM(DS);
    set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')
    order = 10;
    ticstart = tic;
    [A_0_P,timerPE] = S.compute_anchor(order);
    time_PE = time_PE + toc(ticstart);

    order = 7;
    [A_0_outP] = output_anchor(A_0_P,order,epsilon);
    MaxAmp_Aper = [MaxAmp_Aper, max(abs(A_0_outP(index,end-100:end)))];

    %% Exact linear solution, for reference
    omg = DS.Omega;
    g1 = -inv(-omg^2*M+K+1i*omg*C)*fext/(2);
    solution   = @(t) epsilon*2*real(g1*exp(1i*omg*t));
    solution_v = @(t) epsilon*2*real(1i*omg*g1*exp(1i*omg*t));
    Truth  = solution(tspan);
    TruthV = solution_v(tspan);
    MaxAmp_Linear = [MaxAmp_Linear, max(abs(Truth(index,end-100:end)))];

    %% Full-order time integration, for comparison ("True")
    startt = tic;
    q0  = Truth(:,1);
    qd0 = TruthV(:,1);
    tmax = tspan(end)-tspan(1);
    F_ext = @(t) epsilon*Fext_f(t,DS.Omega);
    qdd0 = -M\(F_ext(0)) - K\q0;
    [F, lambda, V, G, DG, G_nl] = functionFromTensors(M, C, K, fnl);

    TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);
    residual = @(q,qd,qdd,t) residual_nonlinear_slow( q, qd, qdd, t, M,K,C,G_nl, F_ext);
    TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
    Fnltime = toc(startt);
    MaxAmp_true = [MaxAmp_true, max(abs(TI_NL.Solution.q(index,end-100:end)))];

end

%% Plot: forced-response curve — full-order vs. exact linear vs. aSSM anchor (periodic & aperiodic)
figure
plot(omegaRange,MaxAmp_true,'-o','color','black')
hold on
plot(omegaRange,MaxAmp_Linear,'-','color','red')
hold on
plot(omegaRange,MaxAmp_Per,'-','color','blue')
hold on
plot(omegaRange,MaxAmp_Aper,'o','color','green')

legend('True','$O(1)$ GSS exact','O(10) GSS periodic','O(15) GSS','Interpreter','latex')
title('$\epsilon = 0.1$','Interpreter','latex')
xlabel('$\Omega$','Interpreter','latex');
ylabel('$|x_5|_{max}$','Interpreter','latex');
