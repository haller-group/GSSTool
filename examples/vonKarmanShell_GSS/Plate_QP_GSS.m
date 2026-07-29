%% Von Karman shell — quasi-periodic forcing: aSSM anchor vs. full-order
% Computes the aSSM anchor trajectory for a von Karman shell (Davenport
% wind-loaded plate model) under quasi-periodic forcing, and compares it
% against a full-order (Newmark) time integration.
%
% For the forcing-weakness ratio r_w, see the companion script
% rw_calculation_QP.m. For a Galerkin-ROM comparison on top of this
% same setup, see Plate_QP_GSS_Galerkin.m.

clear all; close all; clc
run ../../install.m

%% System parameters and finite element model
nDiscretization = 10; % Discretization parameter (#DOFs is proportional to the square of this number)

[M,C,K,fnl,fext,outdof,MyAssembly] = build_model_Davenport(nDiscretization);
n = length(M); % number of degrees of freedom
disp(['Number of degrees of freedom = ' num2str(n)])
disp(['Phase space dimensionality = ' num2str(2*n)])

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')
[V,D,W] = DS.linear_spectral_analysis();

%% Build quasi-periodic forcing
reps = 20;
tspan = linspace(0,2*pi/imag(D(1))*reps,2000);

Omega1 = imag(D(1));
Omega2 = imag(D(5));
sig = sin(Omega1*tspan) + sin(Omega2*tspan);
sig = sin(Omega1*tspan) + sin(pi*Omega1*tspan);

f_0 = fext;
Fext_T = (f_0.* sig);

h = tspan(2);
Fext = Fext_T;
nc_order = 1;

set(DS,'Fext',full(Fext),'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
DS.ncc_order = nc_order;

figure
plot(tspan,sig,'-','LineWidth',3,'color',[0.5 0 0.5])
xlabel('$t$ [s]','Interpreter','latex');
ylabel('$g(t)$','Interpreter','latex');
xlim([0 2*pi/imag(D(1))*reps])

%% Modal decay diagnostics, to inform modal truncation
[VV, dd] = eigs(full(K),full(M),500,'smallestabs');
dd = diag(dd);
[~, ind] = sort(dd);
V = VV(:,ind);
mu = diag(V.' * M * V);
U = V * diag( 1./ sqrt(mu) );
omega0 = sqrt(diag((U.' * K * U)));
zeta = diag((U.' * C * U))./ (2*omega0);
lambda_p = (-zeta + sqrt(zeta.^2 - 1)).* omega0;
lambda_m = (-zeta - sqrt(zeta.^2 - 1)).* omega0;
check_1 = exp(lambda_p*h);
figure
plot(real(check_1),'LineWidth',3,'Color','blue')
hold on
plot(imag(check_1),'LineWidth',3,'Color','red')
xlabel('$Modes$','Interpreter','latex');
ylabel('$exp(\lambda h)$','Interpreter','latex');

% Set modes for modal truncation.
DS.nmodes = 200;

deltat = 4.2641e-04;

[VV, dd] = eig(full(K),full(M));
dd = diag(dd);
[~, ind] = sort(dd);
V = VV(:,ind);

mu = diag(V.' * M * V);
U = V * diag( 1./ sqrt(mu) );
omega0 = sqrt(diag((U.' * K * U)));
zeta = diag((U.' * C * U))./ (2*omega0);

lambda_p = (-zeta + sqrt(zeta.^2 - 1)).* omega0;
check = exp(real(lambda_p)*deltat);

figure
plot(1:1:300,check(1:300),'-','LineWidth',2,'color',[0.4 0.4 0.4]);
xlabel('$j^{th}$ mode','Interpreter','latex');
ylabel('$e^{\delta t \mathrm{Re}\lambda_j}$','Interpreter','latex');
title('$\delta t \approx 10^{-4}$','Interpreter','latex');

%% Compute the aSSM anchor trajectory
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')
order = 4;
ticstart2 = tic;
[A_0_N,timerN] = S.compute_anchor(order);
time_NL = toc(ticstart2);

%% Full-order (Newmark) time integration, for comparison against the anchor
epsilon = 3000;
startt = tic;

order = 4;
[A_0_outP] = output_anchor(A_0_N,order,epsilon);

IC = zeros(2*n,1);
q0  = IC(1:n,1);
qd0 = IC(n+1:2*n,1);
qdd0 = -epsilon*inv(M)*Fext(:,1);
tmax = tspan(end)-tspan(1);

F_ext1 = griddedInterpolant(tspan(:),full(Fext).','linear');
F_ext = @(xa) epsilon*((xa >=tspan(1)) & (xa <=tspan(end))).*F_ext1(xa);

TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005);
residual = @(q,qd,qdd,t) residual_nonlinear_assemble( q, qd, qdd, t, MyAssembly, F_ext);
TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);

%% Plot: full-order trajectory vs. aSSM anchor prediction
figure
epsilon = 3000;
order = 4;
[A_0_outN] = output_anchor(A_0_N,order,epsilon);

indexR  = 990;
indexRd = 990;
plot(TI_NL.Solution.time,TI_NL.Solution.q(indexRd,:),'-','LineWidth',3,'color','black')
hold on
plot(timerN,A_0_outN(indexR,:),'--','LineWidth',3,'color',[0 1 0])
hold on
order = 1;
[A_0_outNL] = output_anchor(A_0_N,order,epsilon);

plot(timerN,A_0_outNL(indexR,:),'--','LineWidth',3,'color',[1 0 0])

xlabel('$t \,[$s$]$','Interpreter','latex');
ylabel('$z_{mid} \,[$m$]$','Interpreter','latex');
title('von Karman shell, qp orbit (all modes),  ($r_w=0.158$)','Interpreter','latex')
xlim([tspan(end-500) tspan(end)])

%% Plot: response torus in 3D (full-order vs. O(1) vs. O(4) aSSM anchor)
zz_axis = 662;
figure
plot3(TI_NL.Solution.q(990,end-1000:end),TI_NL.Solution.qd(990,end-1000:end),TI_NL.Solution.q(zz_axis,end-1000:end),'-','LineWidth',3,'color','black')
hold on
plot3(A_0_outNL(990,end-1000:end),A_0_outNL(990+1320,end-1000:end),A_0_outNL(zz_axis,end-1000:end),'-','LineWidth',3,'color',[0 0 1 0.5])
hold on
plot3(A_0_outN(990,end-1000:end),A_0_outN(990+1320,end-1000:end),A_0_outN(zz_axis,end-1000:end),'-','LineWidth',3,'color',[1 0 0 0.5])
grid on
box on
xlabel('$z_{A} \,[$m$]$','Interpreter','latex');
ylabel('$\dot{z}_{A} \,[$m/s$]$','Interpreter','latex');
zlabel('$\beta_{M} \, [$m$]$','Interpreter','latex');
title('$m = 200$ slowest modes','Interpreter','latex');

legend('Full model','$O(1)$ GSS','$O(4)$ GSS','Interpreter','latex')
view(144,13)

%% NMTE error (normalized mean trajectory error) against the full-order solution
True = [TI_NL.Solution.q(:,end-500:end);TI_NL.Solution.qd(:,end-500:end)];

NMTE_L = sum(sqrt(sum((True - A_0_outNL(:,end-500:end)).^2)))/(500*max(sqrt(sum(True.^2))))

NMTE_NL = sum(sqrt(sum((True - A_0_outN(:,end-500:end)).^2)))/(500*max(sqrt(sum(True.^2))))
