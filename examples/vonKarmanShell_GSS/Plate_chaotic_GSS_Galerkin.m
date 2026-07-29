%% Von Karman shell — chaotic forcing: aSSM anchor vs. Galerkin ROM vs. full-order
% Computes the aSSM anchor trajectory for a von Karman shell (Davenport
% wind-loaded plate model) under chaotically-forced (Rossler-driven)
% excitation, and compares it against both a full-order (Newmark) time
% integration and a Galerkin-projected reduced-order model built on the
% same truncated modal basis used for the GSS reduction.
%
% For the forcing-weakness ratio r_w, see the companion script
% rw_calculation_chaotic.m. For the non-Galerkin comparison, see
% Plate_chaotic_GSS.m.

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

%% Build chaotic forcing (Rossler system, second state variable)
L = 0;
N_Max = 1000;
tspan = linspace(L,100,N_Max);
loren = @(t,y) rossler(t,y);
IC = [0;0.3;0.5];
[tk,yy] = ode45(loren,tspan,IC);

F_a_new = (yy(:,2)/max(abs(yy(:,2)))).';

tspan = (linspace(L,100,N_Max)/100)*0.7;

Force_Lorenz = griddedInterpolant(tspan(:),F_a_new(:),'linear');
check_F = Force_Lorenz(tspan);

sig = F_a_new;
f_0 = fext;
Fext_T = (f_0.* sig);

h = tspan(2);
Fext = Fext_T;
nc_order = 1;

set(DS,'Fext',full(Fext),'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
DS.ncc_order = nc_order;

figure
plot(tspan,sig,'-','LineWidth',3,'color','blue')
hold on
plot(tspan,check_F,'-','LineWidth',3,'color','blue')
xlabel('$t$','Interpreter','latex');
ylabel('$g(t)$','Interpreter','latex');
xlim([0 tspan(end)])

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
DS.nmodes = 150;

%% Galerkin projection basis (slowest DS.nmodes modes, mass-normalized)
% Reuses the mass-normalized eigenvectors U computed above (sorted ascending
% by eigenvalue), truncated to the same number of modes used for the GSS
% reduction, so the Galerkin ROM and the GSS ROM are built on a comparable
% reduced subspace.
Phi = U(:,1:DS.nmodes);
M_gal = Phi.' * M * Phi;
C_gal = Phi.' * C * Phi;

%% Compute the aSSM anchor trajectory
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')
order = 6;
ticstart2 = tic;
[A_0_N,timerN] = S.compute_anchor(order);
time_NL = toc(ticstart2);

%% Full-order (Newmark) time integration, for comparison against the anchor
epsilon = 12000;
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

%% Galerkin-projected ROM integration (slowest DS.nmodes modes)
% Same forcing, same Newmark settings, same ICs as the FOM run above, but
% projected onto the truncated modal basis Phi for direct comparison against
% the GSS reduced dynamics.
q0_gal   = Phi.' * M * q0;
qd0_gal  = Phi.' * M * qd0;
qdd0_gal = Phi.' * M * qdd0;

TI_GAL = ImplicitNewmark('timestep',h,'alpha',0.005);
residual_gal = @(q,qd,qdd,t) residual_galerkin_rom( q, qd, qdd, t, MyAssembly, F_ext, Phi, M_gal, C_gal);
TI_GAL.Integrate(q0_gal,qd0_gal,qdd0_gal,tmax,residual_gal);

% Expand back to full constrained DOFs for plotting against the FOM/GSS results
GAL_q_full  = Phi * TI_GAL.Solution.q;
GAL_qd_full = Phi * TI_GAL.Solution.qd;

%% Plot: full-order vs. Galerkin ROM vs. aSSM anchor prediction
figure
epsilon = 12000;
order = 6;
[A_0_outN] = output_anchor(A_0_N,order,epsilon);

indexR  = 660+1320;
indexRd = 660;
plot(TI_NL.Solution.time,TI_NL.Solution.qd(indexRd,:),'-','LineWidth',3,'color','black')
hold on
plot(timerN,A_0_outN(indexR,:),'--','LineWidth',3,'color',[0 1 0])
hold on
order = 1;
[A_0_outNL] = output_anchor(A_0_N,order,epsilon);

plot(timerN,A_0_outNL(indexR,:),'--','LineWidth',3,'color',[1 0 0])
hold on
plot(TI_GAL.Solution.time,GAL_qd_full(indexRd,:),'-','LineWidth',3,'color',[0.4 0 0.4 0.5])

xlabel('$t \,[$s$]$','Interpreter','latex');
ylabel('$\dot{z}_{mid} \,[$m/s$]$','Interpreter','latex');
title('von Karman shell, chaotic forcing, 150 modes ($r_w=0.22$)','Interpreter','latex')

%% Plot: response torus in 3D, 150 slowest modes (full-order vs. Galerkin ROM vs. O(1)/O(6) aSSM anchor)
figure
plot3(TI_NL.Solution.q(990,end-700:end-1),TI_NL.Solution.qd(990,end-700:end-1),TI_NL.Solution.q(660,end-700:end-1),'-','LineWidth',3,'color','black')
hold on
plot3(A_0_outNL(990,end-700:end-1),A_0_outNL(990+1320,end-700:end-1),A_0_outNL(660,end-700:end-1),'-','LineWidth',3,'color',[0 0 1 0.5])
hold on
plot3(A_0_outN(990,end-700:end-1),A_0_outN(990+1320,end-700:end-1),A_0_outN(660,end-700:end-1),'-','LineWidth',3,'color',[1 0 0 0.8])
hold on
plot3(GAL_q_full(990,end-700:end-1),GAL_qd_full(990,end-700:end-1),GAL_q_full(660,end-700:end-1),'-','LineWidth',3,'color',[0.4 0 0.4 0.5])
grid on
box on
xlabel('$z_{A} \,[$m$]$','Interpreter','latex');
ylabel('$\dot{z}_{A} \,[$m$]$','Interpreter','latex');
zlabel('$z_{M} \, [$m$]$','Interpreter','latex');
title('$m = 150$ slowest modes','Interpreter','latex');

legend('Full model','$O(1)$ GSS','$O(6)$ GSS','Galerkin ROM (150 modes)','Interpreter','latex')
view(144,13)

%% Plot: response torus in 3D, 200 slowest modes (full-order vs. Galerkin ROM vs. O(1)/O(4) aSSM anchor)
zz_axis = 660;
figure
plot3(TI_NL.Solution.q(990,end-1000:end),TI_NL.Solution.qd(990,end-1000:end),TI_NL.Solution.q(zz_axis,end-1000:end),'-','LineWidth',3,'color','black')
hold on
plot3(A_0_outNL(990,end-1000:end),A_0_outNL(990+1320,end-1000:end),A_0_outNL(zz_axis,end-1000:end),'-','LineWidth',2,'color',[0 1 0 0.5])
hold on
plot3(A_0_outN(990,end-1000:end),A_0_outN(990+1320,end-1000:end),A_0_outN(zz_axis,end-1000:end),'-','LineWidth',3,'color',[1 0 0 0.8])
hold on
plot3(GAL_q_full(990,end-1000:end),GAL_qd_full(990,end-1000:end),GAL_q_full(zz_axis,end-1000:end),'-','LineWidth',3,'color',[0.4 0 0.4 0.5])
grid on
box on
xlabel('$z_{A} \,[$m$]$','Interpreter','latex');
ylabel('$\dot{z}_{A} \,[$m/s$]$','Interpreter','latex');
zlabel('$z_{M} \, [$m$]$','Interpreter','latex');
title('$m = 200$ slowest modes','Interpreter','latex');

legend('Full model','$O(1)$ GSS','$O(4)$ GSS','Galerkin ROM (150 modes)','Interpreter','latex')
view(144,13)

%% NMTE error (normalized mean trajectory error) against the full-order solution
True = [TI_NL.Solution.q(:,end-500:end);TI_NL.Solution.qd(:,end-500:end)];

NMTE_L = sum(sqrt(sum((True - A_0_outNL(:,end-500:end)).^2)))/(500*max(sqrt(sum(True.^2))))

NMTE_NL = sum(sqrt(sum((True - A_0_outN(:,end-500:end)).^2)))/(500*max(sqrt(sum(True.^2))))

True_GAL = [GAL_q_full(:,end-500:end);GAL_qd_full(:,end-500:end)];
NMTE_GAL = sum(sqrt(sum((True - True_GAL).^2)))/(500*max(sqrt(sum(True.^2))))
