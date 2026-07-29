clear all; close all; clc
run ../../install.m
%% rw_calculation_vonKarmanBeam_AnchorTool.m
%
% Computes the forcing-weakness ratio r_w for the von Karman beam under
% earthquake ground-motion forcing (companion to vonKarmanBeam_AnchorTool.m),
% following the definition used in:
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
%               <|| C*qd_free + F(q_free) ||>_t
%
% where F(q) is the TOTAL (linear+nonlinear) internal elastic force from
% Assembly.tangent_stiffness_and_force (NOT K*q plus a separate nonlinear
% term - K is only the tangent of F, summing it on top would double count
% the linear part), consistent with the Assembly-based convention used in
% this example (same convention as the plate examples).
%
% This example uses the SAME model and forcing data as
% vonKarmanBeam_AnchorTool.m: a von Karman beam (YetAnotherFEcode finite
% element model) under recorded earthquake ground acceleration
% (RSN28_PARKF_VER.csv / RSN28_PARKF_HOR1.csv), scaled by
% epsilon = 10*Epsilon_Max(2) to match the final forced integration run in
% that script. read_earthquake_accel.m, build_model.m, and
% assemble_force_newmark.m are assumed available on the MATLAB path, as in
% the original script.

%% system parameters

nElements = 20;

%% generate model

[M,C,K,fnl,f_01,outdof,MyAssembly,V0,v1] = build_model(nElements);
n = length(M);

DS = DynamicalSystem();
set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')
DS.order = 2;

%% Load earthquake forcing (must match vonKarmanBeam_AnchorTool.m exactly)
dtt = 0.01;
g = 9.8;
stringV = 'RSN28_PARKF_VER.csv';
stringH = 'RSN28_PARKF_HOR1.csv';

[TV,TH,tspan] = read_earthquake_accel(stringV,stringH,dtt,g);
Epsilon_Max = [max(TV),max(TH)];

TV = TV/max(TH);
TH = TH/max(TH);

vecAxial = [1 0 0];
vecTransv = [0 1 0];
FAr = repmat(vecAxial,1,nElements);
FTr = repmat(vecTransv,1,nElements);

FV_axial = griddedInterpolant(tspan(:),TV(:),'linear');
FH_transverse = griddedInterpolant(tspan(:),TH(:),'linear');

tfinal = tspan(end);

% epsilon used for the final forced run in vonKarmanBeam_AnchorTool.m
epsilon = Epsilon_Max(2);

% Assembled forcing time series over tspan (force units, already includes
% the M-weighting baked into assemble_force_newmark)
F_ext = @(t) assemble_force_newmark(epsilon, ...
    (((t >=0) & (t <=tfinal)).*FH_transverse(t)), ...
    (((t >=0) & (t <=tfinal)).*FV_axial(t)), ...
    FAr.', FTr.', M);

Fext_hist = zeros(n,length(tspan));
for ind = 1:length(tspan)
    Fext_hist(:,ind) = F_ext(tspan(ind));
end

%% Random-IC ensemble setup
% Random initial conditions are drawn as random linear combinations of the
% slowest nmodes_rw mass-normalized eigenmodes of (K,M), normalized to a
% fixed displacement amplitude q0_amp. No precedent IC convention or rw
% block was present in vonKarmanBeam_AnchorTool.m, so q0_amp is a judgment
% call - check it against this beam's actual displacement scale (e.g.
% compare to typical |q| values from the forced KI_NL run in the original
% script) before trusting the resulting r_w value.

rng(0); % fixed seed for reproducibility of the random IC ensemble

n_IC = 5;        % number of random initial conditions to average over
nmodes_rw = 10;   % number of slow modes spanning the random IC subspace
q0_amp = 0.005;    % displacement amplitude - not verified against this model's natural scale

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

decay_tol = 0.9;              % stop once amplitude has decayed to this fraction of initial
safety_factor = 3;              % extra margin beyond the predicted decay time
tmax_k = safety_factor * log(1/decay_tol) / decay_rate_min;  % per-IC or just use decay_rate_min once outside loop



h = dtt;

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
F_e = sqrt(sum(Fext_hist.^2));

rw = (trapz(tSP1,F_e)) / F_net_avg;

disp(' ')
disp(['r_w (von Karman beam, earthquake forcing, averaged over ' num2str(n_IC) ' random ICs) = ' num2str(rw)])
