clear all; close all; clc
run ../../install.m
%% rw_calculation_OscillatorChain.m
%
% Computes the forcing-weakness ratio r_w for the oscillator chain under
% the recorded/data-driven forcing signal used in
% Oscillator_chain_LTSM_compare_GSS.m (Test3 dataset), following the
% definition used in:
%
%   G. Haller & R.S. Kaundinya, "Nonlinear Model Reduction to Temporally
%   Aperiodic Spectral Submanifolds", Chaos 34 (2024) 043152.
%   https://arxiv.org/abs/2404.05355
%
% r_w quantifies how weak the applied forcing is relative to the system's
% own intrinsic (nonlinear) restoring force, evaluated along a free
% (autonomous, unforced) trajectory of the same system:
%
%               <|| epsilon * Fext(t) ||>_t
%        r_w =  ---------------------------
%               <|| C*qd_free + K*q_free + G_nl([q_free;qd_free]) ||>_t
%
% Here G_nl is the NONLINEAR-ONLY correction term returned by
% functionFromTensors (same convention as AxialMovingBeam_chirp_GSS.m):
% K*q + G_nl([q;qd]) is the correct TOTAL restoring force in this
% representation.
%
% NOTE: Oscillator_chain_LTSM_compare_GSS.m has no precedent rw block or
% autonomous-trajectory IC convention to follow, unlike the plate and
% axial beam examples. The autonomous IC below (perturbation along the
% slowest eigenmode, displacement amplitude q0_amp) is therefore a
% reasonable but not verified choice - check q0_amp against the
% displacement scale this model actually operates at (e.g. against
% TI_NL.Solution.q amplitudes in the original forced run) before trusting
% the resulting r_w value.

%% Example setup (must match Oscillator_chain_LTSM_compare_GSS.m exactly)

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

%% Load forcing data (Test3 dataset, must match Oscillator_chain_LTSM_compare_GSS.m exactly)

f_0_1 = zeros(n,1);
f_0_1(1) = 1;

f_0_2 = zeros(n,1);
f_0_2(end) = 1;

Test3 = load('0.50_oscillatorchain_nElements_20_latdim_9_test3.mat');
F1 = Test3.force_orig;

Fext_T = f_0_1*(-F1(:,1).') + f_0_2*(-F1(:,2).');

tspan = Test3.t;
h = 0.01;
tfinal = tspan(end);
Fext = Fext_T;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);

% Resample onto the finer grid used for the production integration in
% Oscillator_chain_LTSM_compare_GSS.m
tspany = linspace(0,100,30001);
F_ext1 = griddedInterpolant(DS.tforce(:),DS.Fext.','linear');
F_ext_full = @(xa) (((xa >=DS.tforce(1)) & (xa <=DS.tforce(end))).*(F_ext1(xa)).');

Fext_resampled = F_ext_full(tspany);
hy = tspany(2);
set(DS,'Fext',Fext_resampled,'tforce',tspany,'tend',tspany(end)-tspany(1),'dh',hy);

epsilon = 1.8; % must match the epsilon used for the production run in
               % Oscillator_chain_LTSM_compare_GSS.m (NOT the earlier
               % epsilon = 0.1 used only for noise-signal scaling)

[F, lambda, V, G, DG, G_nl] = functionFromTensors(M, C, K, fnl);

%% Random-IC ensemble setup
% Random initial conditions are drawn as random linear combinations of the
% slowest nmodes_rw mass-normalized eigenmodes of (K,M), normalized to a
% fixed displacement amplitude q0_amp. No precedent IC convention or rw
% block exists in Oscillator_chain_LTSM_compare_GSS.m, so q0_amp is a
% judgment call - check it against this model's actual displacement scale
% (e.g. against TI_NL.Solution.q amplitudes in the original forced run)
% before trusting the resulting r_w value.

rng(0); % fixed seed for reproducibility of the random IC ensemble

n_IC = 1;        % number of random initial conditions to average over
nmodes_rw = 10;   % number of slow modes spanning the random IC subspace
q0_amp = 0.5;     % displacement amplitude - not verified against this model's natural scale

[VV_rw, dd_rw] = eigs(full(K),full(M),nmodes_rw,'smallestabs');
dd_rw = diag(dd_rw);
[~, ind_rw] = sort(dd_rw);
V_rw = VV_rw(:,ind_rw);
mu_rw = diag(V_rw.' * M * V_rw);
U_rw = V_rw * diag( 1./ sqrt(mu_rw) ); % mass-normalized slow modes for random IC subspace

% Decay-rate convergence check for the sampled mode subspace
omega0_rw = sqrt(dd_rw(ind_rw(1:nmodes_rw)));
zeta_rw   = diag(U_rw.' * C * U_rw) ./ (2*omega0_rw);
decay_rate_rw = zeta_rw .* omega0_rw;
decay_rate_min = min(decay_rate_rw);

decay_tol = 1e-3;              % stop once amplitude has decayed to this fraction of initial
safety_factor = 3;              % extra margin beyond the predicted decay time
tmax_k = safety_factor * log(1/decay_tol) / decay_rate_min;  % per-IC or just use decay_rate_min once outside loop


F_net_avg_list = zeros(n_IC,1);

for k = 1:n_IC

    coeffs = randn(nmodes_rw,1);
    q0 = U_rw * coeffs;
    q0 = q0_amp * q0 / max(abs(q0));
    qd0 = zeros(n,1);
    qdd0 = -K\q0;

    TI_NL1 = ImplicitNewmark('timestep',hy,'alpha',0.005,'RelTol',1e-05,'linear',false);
    residual_uf = @(q,qd,qdd,t) residual_nonlinear_uf( q, qd, qdd, t, M, K, C, G_nl);
    TI_NL1.Integrate(q0,qd0,qdd0,tmax_k,residual_uf);

    Inter_Forces = C*TI_NL1.Solution.qd + K*TI_NL1.Solution.q + G_nl([TI_NL1.Solution.q;TI_NL1.Solution.qd]);
    F_net = sqrt(sum(Inter_Forces.^2));
    tSP2 = TI_NL1.Solution.time;

    F_net_avg_list(k) = trapz(tSP2,F_net);

    disp(['  IC ' num2str(k) '/' num2str(n_IC) ': time-avg ||C*qd+K*q+G_nl|| = ' num2str(F_net_avg_list(k))])
end

F_net_avg = mean(F_net_avg_list);

%% Compute rw

tSP1 = tspany;
F_e = epsilon*sqrt(sum(Fext_resampled.^2));

rw = (trapz(tSP1,F_e)) / F_net_avg;

disp(' ')
disp(['r_w (oscillator chain, Test3 forcing, averaged over ' num2str(n_IC) ' random ICs) = ' num2str(rw)])
