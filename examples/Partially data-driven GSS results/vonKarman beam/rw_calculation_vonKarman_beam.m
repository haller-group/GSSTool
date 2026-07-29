clear all; close all; clc

%% rw_calculation_vonKarman_beam.m
%
% Computes the forcing-weakness ratio r_w for the von Karman beam example
% (companion to vonKarman_beam_SSM_GSS.m), following the definition used
% in:
%
%   G. Haller & R.S. Kaundinya, "Nonlinear Model Reduction to Temporally
%   Aperiodic Spectral Submanifolds", Chaos 34 (2024) 043152.
%   https://arxiv.org/abs/2404.05355
%
% r_w quantifies how weak the applied forcing is relative to the system's
% own intrinsic (nonlinear) restoring force, evaluated along free
% (autonomous, unforced) trajectories of the same system:
%
%               <|| epsilon * Fext(t) ||>_t
%        r_w =  ---------------------------------
%               < <|| C*qd_free + F(q_free) ||>_t >_IC
%
% where:
%   - F(q) is the TOTAL (linear + nonlinear) internal elastic force from
%     Assembly.tangent_stiffness_and_force, NOT K*q + nonlinear correction
%     (K is only the tangent/Jacobian of F, not a separate additive term -
%     adding K*q on top of F(q) double-counts the linear part).
%   - <.>_t denotes the time average (trapz(.)/max(t)).
%   - <.>_IC denotes an average over multiple random free-trajectory
%     realizations, each started from a random initial condition in the
%     (reduced) phase space, rather than relying on a single perturbation
%     direction.
%
% Uses the same model and forcing history as vonKarman_beam_SSM_GSS.m,
% loaded from SSM_details_forcing_realiation2.mat (Forcing_history), with
% the model built via build_modelvk.m.

%% system parameters
nElements = 20;

%% generate model
[M,C,K,fnl,f_01,outdof,MyAssembly,V0,v1] = build_modelvk(nElements);
n = length(M);

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')
DS.order = 2;

%% Load forcing history (must match vonKarman_beam_SSM_GSS.m exactly)
T0 = 200; % PSD frequency domain resolution is ~ 1/T0
nPoints = 2^13*1; % controls the accuracy of the numerical differential equation
tspan = linspace(0,T0,nPoints+1);
tspan = tspan.';
load('SSM_details_forcing_realiation2.mat','R0','W0','S','Forcing_history','Xwq')

%% Random-IC ensemble setup
% Random initial conditions are drawn as random linear combinations of the
% slowest nmodes_rw mass-normalized eigenmodes of (K,M), normalized to a
% fixed displacement amplitude q0_amp. No precedent IC convention or rw
% block was present in the original earthquake-forced beam script, so
% q0_amp is a judgment call - check it against this beam's actual
% displacement scale before trusting the resulting r_w value.

rng(0); % fixed seed for reproducibility of the random IC ensemble

n_IC = 5;         % number of random initial conditions to average over
nmodes_rw = 10;   % number of slow modes spanning the random IC subspace
q0_amp = 0.0005;  % displacement amplitude - not verified against this model's natural scale

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

decay_tol = 1e-1;      % stop once amplitude has decayed to this fraction of initial
safety_factor = 3;     % extra margin beyond the predicted decay time
tmax_k = safety_factor * log(1/decay_tol) / decay_rate_min;

h = tspan(2);

F_net_avg_list = zeros(n_IC,1);

for k = 1:n_IC

    coeffs = randn(nmodes_rw,1);
    q0 = U_rw * coeffs;
    q0 = q0_amp * q0 / max(abs(q0));
    qd0 = zeros(n,1);
    qdd0 = -K\q0;

    TI_NL1 = ImplicitNewmark('timestep',h,'alpha',0.005);
    residual_uf = @(q,qd,qdd,t) residual_nonlinear_assemble_uf( q, qd, qdd, t, MyAssembly);
    TI_NL1.Integrate(q0,qd0,qdd0,tmax_k,residual_uf);

    n_samples = size(TI_NL1.Solution.q,2);
    F_elastic_hist = zeros(n,n_samples);
    for ind = 1:n_samples
        u_ind = MyAssembly.unconstrain_vector(TI_NL1.Solution.q(:,ind));
        [~, F_ind] = MyAssembly.tangent_stiffness_and_force(u_ind);
        F_elastic_hist(:,ind) = MyAssembly.constrain_vector(F_ind);
    end

    Inter_Forces = C*TI_NL1.Solution.qd + F_elastic_hist;
    F_net = sqrt(sum(Inter_Forces.^2));
    tSP2 = TI_NL1.Solution.time;

    F_net_avg_list(k) = trapz(tSP2,F_net);

    disp(['  IC ' num2str(k) '/' num2str(n_IC) ': time-avg ||C*qd+F(q)|| = ' num2str(F_net_avg_list(k))])
end

F_net_avg = mean(F_net_avg_list);

%% Compute rw
tSP1 = tspan;
F_e = sqrt(sum(Forcing_history.^2));

rw = (trapz(tSP1,F_e)) / F_net_avg;

disp(' ')
disp(['r_w (von Karman beam, averaged over ' num2str(n_IC) ' random ICs) = ' num2str(rw)])
