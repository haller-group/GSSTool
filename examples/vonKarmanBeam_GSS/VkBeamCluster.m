%% Geometrically nonlinear von Karman beam — cluster sweep over mesh size
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
% Runs the earthquake-forced aSSM anchor computation (piecewise-exact and
% Newmark solves) together with a full-order Newmark time integration, for
% a sweep of mesh resolutions, in parallel on a cluster. Results for each
% mesh size are saved to outputSSR<i>.mat / outputFOM<i>.mat.

clear all; close all; clc
run ../../install.m
exampleDir = fileparts(mfilename('fullpath'));
cd(exampleDir)

%% System parameters
nElements_array = [20,40,60,100,300,600,1800,5400];

euler = parcluster('local');
pool = parpool(euler);

parfor iger = 1:max(size(nElements_array))

    nElements = nElements_array(iger);
    [M,C,K,fnl,f_01,outdof,MyAssembly] = build_model(nElements);
    n = length(M); % DOFs per node: axial, transverse, angle (from clamped end)
    Specify_dof = n-1;

    DS = DynamicalSystem();
    set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
    set(DS.Options,'Emax',10,'Nmax',10,'notation','multiindex')

    dtt = 0.01;
    g = 9.8;
    stringV = 'RSN28_PARKF_VER.csv';
    stringH = 'RSN28_PARKF_HOR1.csv';
    [TV,TH,tspan] = read_earthquake_accel(stringV,stringH,dtt,g);
    Epsilon_Max = [max(TV),max(TH)];
    TV = TV/max(TH);
    TH = TH/max(TH);

    FV_axial      = griddedInterpolant(tspan(:),TV(:),'linear');
    FH_transverse = griddedInterpolant(tspan(:),TH(:),'linear');
    vecAxial  = [1 0 0];
    vecTransv = [0 1 0];
    FAr = repmat(vecAxial,1,nElements);
    FTr = repmat(vecTransv,1,nElements);
    Fext = M*(TH.'.*FTr.' + TV.'.*FAr.');
    tfinal = tspan(end);

    nc_order = 1;
    set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',(tspan(2)-tspan(1))/nc_order);
    DS.ncc_order = nc_order;
    epsilon = 1.5*Epsilon_Max(2);

    %% aSSM anchor: piecewise-exact solve
    S = SSM(DS);
    set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Second_order_piecewise_exact')
    order = 5;
    ticstart = tic;
    [A_0_P,timerPE,tn0] = S.compute_anchor(order);
    order = 1;
    [A_0_outPL] = output_anchor_dof(A_0_P,order,epsilon,Specify_dof);
    order = 5;
    [A_0_outP] = output_anchor_dof(A_0_P,order,epsilon,Specify_dof);
    time_PE = toc(ticstart);

    %% aSSM anchor: Newmark solve
    S = SSM(DS);
    set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Newmark')
    order = 5;
    ticstart2 = tic;
    [A_0_N,timerN,tnu] = S.compute_anchor(order);
    order = 1;
    [A_0_outNL] = output_anchor_dof(A_0_N,order,epsilon,Specify_dof);
    order = 5;
    [A_0_outN] = output_anchor_dof(A_0_N,order,epsilon,Specify_dof);
    time_NL = toc(ticstart2);

    parsaveAnchorSol(sprintf('outputSSR%d.mat', iger), time_NL,time_PE,timerPE,timerN,A_0_outN,A_0_outNL,A_0_outP,A_0_outPL,n,tn0);

    %% Full-order (Newmark) time integration, for comparison
    startt = tic;
    h = 0.001;
    IC = zeros(2*n,1);
    q0  = IC(1:n);
    qd0 = IC(n+1:2*n);
    tmax = tspan(end)-tspan(1);
    F_ext = @(t) assemble_force_newmark(epsilon,(((t >=0) & (t <=tfinal)).*FH_transverse(t)),(((t >=0) & (t <=tfinal)).*FV_axial(t)),FAr.',FTr.',M);
    qdd0 = -M\(F_ext(0));
    TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-04,'ATS',true);

    residual = @(q,qd,qdd,t) residual_nonlinear_assemble( q, qd, qdd, t, MyAssembly, F_ext);
    TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
    Final_sol = TI_NL.Solution.q(Specify_dof,:);
    Fnltime = toc(startt);
    parsaveFullorderModel(sprintf('outputFOM%d.mat', iger), Fnltime,Final_sol,n);

end
pool.delete()


function parsaveAnchorSol(fname, time_NL,time_PE,timerPE,timerN,A_0_outN,A_0_outNL,A_0_outP,A_0_outPL,n,tn0)
save(fname, 'time_NL','time_PE','timerPE','timerN','A_0_outN','A_0_outNL','A_0_outP','A_0_outPL','n','tn0','-v7.3')
end

function parsaveFullorderModel(fname, Fnltime,Final_sol,n)
save(fname, 'Fnltime','Final_sol','n','-v7.3')
end
