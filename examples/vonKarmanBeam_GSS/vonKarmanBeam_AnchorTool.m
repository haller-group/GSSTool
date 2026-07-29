%% Geometrically nonlinear von Karman beam — earthquake-forced aSSM anchor trajectory
% Finite element model from:
%   Jain, S., Tiso, P., & Haller, G. (2018). Exact nonlinear model reduction
%   for a von Karman beam: slow-fast decomposition and spectral submanifolds.
%   Journal of Sound and Vibration, 423, 195-211.
%   https://doi.org/10.1016/J.JSV.2018.01.049
%
% Finite element code from:
%   Jain, S., Marconi, J., Tiso P. (2020). YetAnotherFEcode (v1.1). Zenodo.
%   http://doi.org/10.5281/zenodo.4011282
%
% Computes the aSSM anchor trajectory x*(t) for a cantilevered von Karman
% beam under recorded earthquake ground acceleration, and compares it
% against a full-order (Newmark) time integration.

clear all; close all; clc
run ../../install.m
exampleDir = fileparts(mfilename('fullpath'));
cd(exampleDir)

%% System parameters and finite element model
nElements = 20;
[M,C,K,fnl,f_01,outdof,MyAssembly,V0,v1] = build_model(nElements);
n = length(M); % DOFs per node: axial, transverse, angle (from clamped end)

%% Dynamical system setup
% Forced system:
%   M*xddot + C*xdot + K*x + f(x,xdot) = epsilon*f_ext(Omega*t)
% First-order form:
%   B*zdot = A*z + F(z) + epsilon*F_ext(phi),   phidot = Omega
DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')
DS.order = 2;

%% Earthquake ground-motion forcing
dtt = 0.01;
g = 9.8;
stringV = 'RSN28_PARKF_VER.csv';
stringH = 'RSN28_PARKF_HOR1.csv';

[TV,TH,tspan] = read_earthquake_accel(stringV,stringH,dtt,g);
Epsilon_Max = [max(TV),max(TH)];

% normalize records; forcing below is transverse-only (axial term zeroed)
TV = TV/max(TH);
TH = TH/max(TH);

FV_axial      = griddedInterpolant(tspan(:),TV(:),'linear');
FH_transverse = griddedInterpolant(tspan(:),TH(:),'linear');
vecAxial  = [1 0 0];
vecTransv = [0 1 0];
FAr = repmat(vecAxial,1,nElements);
FTr = repmat(vecTransv,1,nElements);
Fext = M*(TH.'.*FTr.' + TV.'.*FAr.'*0);

tfinal = tspan(end);
nc_order = 1;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',0.001);
DS.ncc_order = nc_order;

%% Linear modal decomposition and time-propagation matrix for the anchor solver
nmode = 6;
Lin_A = DS.BinvA;

[V, lambda, W] = eig(full(DS.A),full(DS.B));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) );  % mass-normalized right modes
DS.W = (W*diag(1./(sqrt(mu)')));  % mass-normalized left modes
DS.lambda = diag(lambda);

% sorted eigendecomposition of Lin_A, truncated to the nmode slowest modes,
% used to build the exponential time-propagator DS.Timelin
[V,D] = eig(full(Lin_A));
[~,ind] = sort(diag(-real(D)));
Ds = D(ind,ind);
Vs = V(:,ind);
ds = diag(Ds);
ds = ds(1:nmode);
V_sub = Vs(:,1:nmode);

Eval_vec = sparse(exp(ds.*0.001));
VI = Vs\speye(2*n,2*n);
VI_sub = sparse(VI(1:nmode,:));
V_sub = sparse(V_sub);

Mat_A = cell(1,1);
Mat_A{1,1} = real(sparse((V_sub*(Eval_vec.*VI_sub))));
DS.Timelin = Mat_A;

%% Compute the aSSM anchor trajectory
% Piecewise-exact solve
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex', ...
    'Solve_Method','Second_order_piecewise_exact','Ftype','aperiodic')
order = 5;
ticstart = tic;
[A_0_P,timerPE] = S.compute_anchor(order);
time_PE = toc(ticstart);

% Newmark solve
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Newmark')
order = 10;
ticstart2 = tic;
[A_0_N,timerN] = S.compute_anchor(order);
time_NL = toc(ticstart2);

%% Full-order (Newmark) time integration, for comparison against the anchor
epsilon = 1.5*Epsilon_Max(2);
startt = tic;

h = 0.001;
IC = zeros(2*n,1);
q0  = IC(1:n);
qd0 = IC(n+1:2*n);
tmax = tspan(end)-tspan(1);
F_ext = @(t) assemble_force_newmark(epsilon, ...
    (((t >=0) & (t <=tfinal)).*FH_transverse(t)), ...
    (((t >=0) & (t <=tfinal)).*FV_axial(t)), FAr.', FTr.', M);
qdd0 = -M\(F_ext(0)) - K\q0;

KI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05);
KI_NL.epsilon = epsilon;
residual = @(q,qd,qdd,t) residual_nonlinear_assemble( q, qd, qdd, t, MyAssembly, F_ext);
KI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);

%% Plot: full-order trajectory vs. aSSM anchor prediction
figure
order = 10;
[A_0_outP] = output_anchor(A_0_N,order,epsilon);
order = 1;
[A_0_outPL] = output_anchor(A_0_N,order,epsilon);

indexR  = n-2;
indexRd = n-2;

plot(KI_NL.Solution.time,KI_NL.Solution.q(indexRd,:),'-','LineWidth',3,'color',[0 0 0])
hold on
plot(timerN,A_0_outP(indexR,:),'--','LineWidth',3,'color',[1 0 0])
hold on
plot(timerN,A_0_outPL(indexR,:),'--','LineWidth',3,'color',[0 1 0])

xlabel('$t \,[$s$]$','Interpreter','latex');
ylabel('P_{axial}$','Interpreter','latex');
legend('Full model (time integration)','$O(10)$ GSS $x^*(t)$','$O(1)$ GSS $x^*(t)$','Interpreter','latex')
xlim([0 tfinal])
