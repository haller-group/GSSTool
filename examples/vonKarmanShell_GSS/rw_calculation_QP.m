clear all; close all; clc
run ../../install.m
%% rw_calculation_QP.m
%
% Computes the forcing-weakness ratio r_w for the quasi-periodically forced
% von Karman plate example (companion to Plate_QP_GSS_Galerkin.m),
% following the definition used in:
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
%     direction. This better samples the system's intrinsic restoring
%     force magnitude across the relevant region of phase space, rather
%     than along one specific (e.g. slowest-mode) direction.
%
% NOTE ON SOURCE: this follows the physical definition of r_w as a ratio
% of forcing magnitude to intrinsic restoring-force magnitude, and
% generalizes the single-IC version (previously used in this codebase) to
% an ensemble average over random ICs, per user request. This has not been
% verified against the exact sampling scheme in the companion
% haller-group/Aperiodic-SSMs GitHub repository (which could not be
% directly inspected here) - if you have access to that source, it is
% worth cross-checking the IC sampling distribution/amplitude against
% what's implemented below.

%% system parameters

nDiscretization = 10; % Discretization parameter (#DOFs is proportional to the square of this number)

%% generate model

[M,C,K,fnl,fext,outdof,MyAssembly] = build_model_Davenport(nDiscretization);
n = length(M); % number of degrees of freedom
disp(['Number of degrees of freedom = ' num2str(n)])
disp(['Phase space dimensionality = ' num2str(2*n)])

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')
[V,D,W] = DS.linear_spectral_analysis();

%% Build quasiperiodic forcing (must match Plate_QP_GSS_Galerkin.m exactly)
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

set(DS,'Fext',full(Fext),'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);

epsilon = 3000; % must match the epsilon used in Plate_QP_GSS_Galerkin.m

%% Random-IC ensemble setup
% Random initial conditions are drawn as random linear combinations of the
% slowest nmodes_rw mass-normalized eigenmodes, normalized to a fixed
% amplitude q0_amp (chosen to match the order of magnitude of the
% single-mode IC used previously, 2*max(|imag(W(1:end/2,1))|), so this is
% a like-for-like generalization rather than an arbitrary new scale).
% Increase n_IC for a smoother average at the cost of runtime (each
% realization requires a full nonlinear time integration).

rng(0); % fixed seed for reproducibility of the random IC ensemble

n_IC = 1;              % number of random initial conditions to average over
nmodes_rw = 20;         % number of slow modes spanning the random IC subspace
q0_amp = 0.05;
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

decay_tol = 0.5;              % stop once amplitude has decayed to this fraction of initial
safety_factor = 3;              % extra margin beyond the predicted decay time
tmax_k = safety_factor * log(1/decay_tol) / decay_rate_min;  % per-IC or just use decay_rate_min once outside loop


F_net_avg_list = zeros(n_IC,1);

for k = 1:n_IC

    % Random combination of the slow modes, normalized to unit norm, then
    % scaled to the chosen displacement amplitude.
    coeffs = randn(nmodes_rw,1);
    q0 = U_rw * coeffs;
    q0 = q0_amp * q0 / max(abs(q0));
    qd0 = zeros(n,1);
    qdd0 = -DS.K\q0; % consistent initial acceleration, no external force

    TI_NL1 = ImplicitNewmark('timestep',h,'alpha',0.005);
    residual_uf = @(q,qd,qdd,t) residual_nonlinear_assemble_uf( q, qd, qdd, t, MyAssembly);
    TI_NL1.Integrate(q0,qd0,qdd0,tmax_k,residual_uf);

    % Recompute the TOTAL nonlinear internal force F(q) at every sample
    % along this free trajectory (F already includes the linear part - do
    % NOT add K*q on top of it).
    n_samples = size(TI_NL1.Solution.q,2);
    F_elastic_hist = zeros(n,n_samples);
    for ind = 1:n_samples
        u_ind = MyAssembly.unconstrain_vector(TI_NL1.Solution.q(:,ind));
        [~, F_ind] = MyAssembly.tangent_stiffness_and_force(u_ind);
        F_elastic_hist(:,ind) = MyAssembly.constrain_vector(F_ind);
    end

    Inter_Forces = DS.C*TI_NL1.Solution.qd + F_elastic_hist;
    F_net = sqrt(sum(Inter_Forces.^2));
    tSP2 = TI_NL1.Solution.time;

    F_net_avg_list(k) = trapz(tSP2,F_net);

    disp(['  IC ' num2str(k) '/' num2str(n_IC) ': time-avg ||C*qd+F(q)|| = ' num2str(F_net_avg_list(k))])
end

F_net_avg = mean(F_net_avg_list);

%% Compute rw

tSP1 = tspan;
F_e = epsilon*sqrt(sum(Fext_T.^2));

rw = (trapz(tSP1,F_e)) / F_net_avg;

disp(' ')
disp(['r_w (quasi-periodic forcing case, averaged over ' num2str(n_IC) ' random ICs) = ' num2str(rw)])

