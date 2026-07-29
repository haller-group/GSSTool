clear all; close all; clc
%% rw_calculation_AxialMovingBeam_chirp.m
%
% Computes the forcing-weakness ratio r_w for the axially moving beam under
% chirp base excitation (companion to AxialMovingBeam_chirp_GSS.m),
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
%               <|| C*qd_free + K*q_free + G_nl([q_free;qd_free]) ||>_t
%
% Here G_nl is the NONLINEAR-ONLY correction term returned by
% functionFromTensors (distinct from the Assembly-based examples, where
% the tangent_stiffness_and_force output already includes the linear
% part). K*q + G_nl([q;qd]) is therefore the correct TOTAL restoring force
% in this representation, matching the Inter_Forces convention used
% directly in AxialMovingBeam_chirp_GSS.m.
%
% This follows the SAME autonomous-trajectory convention already present
% in AxialMovingBeam_chirp_GSS.m: perturb along real(W(1:n,1)) (the
% slowest mode of the gyroscopic system), qd0 = 0, qdd0 = -K\q0.

%% Setup Dynamical System (must match AxialMovingBeam_chirp_GSS.m exactly)

n = 10;
[mass,damp,gyro,stiff,fnl,fext] = build_model(n,'nonlinear_damp');

DS = DynamicalSystem();
set(DS,'M',mass,'C',damp+gyro,'K',stiff,'fnl',fnl);
set(DS.Options,'Emax',6,'Nmax',10,'notation','multiindex');
set(DS.Options,'RayleighDamping',false,'BaseExcitation',true);
[V,D,W] = DS.linear_spectral_analysis();

[V, lambda, W] = eig(full(DS.A),full(DS.B));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) );
DS.W = (W*diag(1./(sqrt(mu)')));
DS.lambda = diag(lambda);

%% Forcing function (must match AxialMovingBeam_chirp_GSS.m exactly)
tspan = linspace(0,10,1000);
c0 = 1;
w0 = imag(D(1));
sig = sin(2*pi*(c0/2*tspan.^2+w0*tspan));
f_0 = fext;
Fext_T = (f_0.* sig);

h = tspan(2);
Fext = Fext_T;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);

epsilon = 1; % must match the epsilon used for the final integration run
             % in AxialMovingBeam_chirp_GSS.m

[F, lambda, V, G, DG, G_nl] = functionFromTensors(DS.M, DS.C, DS.K, fnl);

%% Random-IC ensemble setup
% Random initial conditions are drawn as random linear combinations of the
% slowest nmodes_rw STATE-SPACE eigenvectors (columns of V, each a 2n x 1
% vector [q;qd]) from the [V,lambda,W]=eig(DS.A,DS.B) decomposition already
% computed above. A state-space (rather than (K,M)-only) eigenbasis is
% used here because this system is gyroscopic (C = damp+gyro is not
% symmetric), so the displacement/velocity split does not decouple the way
% it does for symmetric (K,M) systems - sampling directly in the state
% space avoids assuming a decoupling that may not hold.
%
% Each random IC is normalized to a displacement amplitude q0_amp matching
% the order of magnitude of the original single-mode IC,
% max(|real(W(1:n,1))|), for a like-for-like comparison.

rng(0); % fixed seed for reproducibility of the random IC ensemble

n_IC = 10;        % number of random initial conditions to average over
nmodes_rw = 6;    % number of slow state-space eigenvectors spanning the random IC subspace
q0_amp = max(abs(real(W(1:n,1)))); % amplitude scale, consistent with prior single-mode IC

% Sort state-space eigenvalues by slowest decay (least negative real part)
[~, ind_rw] = sort(real(lambda),'descend');
V_slow = V(:,ind_rw(1:nmodes_rw));
lambda_slow = lambda(ind_rw(1:nmodes_rw));
decay_rate_min = min(abs(real(lambda_slow))); % slowest-decaying sampled mode sets convergence time
 

decay_tol = 1e-3;              % stop once amplitude has decayed to this fraction of initial
safety_factor = 3;              % extra margin beyond the predicted decay time
tmax_k = safety_factor * log(1/decay_tol) / decay_rate_min;  % per-IC or just use decay_rate_min once outside loop

% tmax = tspan(end)-tspan(1);

F_net_avg_list = zeros(n_IC,1);

for k = 1:n_IC

    coeffs = randn(nmodes_rw,1);
    z0 = real(V_slow * coeffs);
    q0 = z0(1:n);
    qd0 = z0(n+1:end);

    % Normalize to the target displacement amplitude
    if max(abs(q0)) > 0
        scale = q0_amp / max(abs(q0));
        q0 = scale*q0;
        qd0 = scale*qd0;
    end

    qdd0 = -DS.K\q0;

    TI_NL1 = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);
    residual_uf = @(q,qd,qdd,t) residual_nonlinear_uf( q, qd, qdd, t, DS.M, DS.K, DS.C, G_nl);
    TI_NL1.Integrate(q0,qd0,qdd0,tmax_k,residual_uf);

    Inter_Forces = DS.C*TI_NL1.Solution.qd + DS.K*TI_NL1.Solution.q + G_nl([TI_NL1.Solution.q;TI_NL1.Solution.qd]);
    F_net = sqrt(sum(Inter_Forces.^2));
    tSP2 = TI_NL1.Solution.time;

    F_net_avg_list(k) = trapz(tSP2,F_net);

    disp(['  IC ' num2str(k) '/' num2str(n_IC) ': time-avg ||C*qd+K*q+G_nl|| = ' num2str(F_net_avg_list(k))])
end

F_net_avg = mean(F_net_avg_list);

%% Compute rw

tSP1 = tspan;
F_e = epsilon*sqrt(sum(Fext_T.^2));

rw = trapz(tSP1,F_e) / F_net_avg;

disp(' ')
disp(['r_w (axially moving beam, chirp forcing, averaged over ' num2str(n_IC) ' random ICs) = ' num2str(rw)])
