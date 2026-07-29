%% Axially moving beam — chirp forced response and aSSM anchor trajectory
% We consider an axially moving beam under 1:3 internal resonance between
% the first two bending modes. The equation of motion is
%
%   uddot + (C+G)*udot + f(u,udot) = epsilon*Omega^2*g*cos(Omega*t)
%
% where G' = -G is a gyroscopic matrix. With a viscoelastic material
% model the system has nonlinear damping. The beam is subject to base
% excitation whose amplitude depends on the (time-varying, chirped)
% forcing frequency.
%
% This script computes the aSSM anchor trajectory under chirp base
% excitation, compares it against a full-order time integration, renders
% the beam's spatial deflection w(x,t) as a movie, and finishes with a
% single-IC forcing-weakness ratio r_w check (see
% rw_calculation_AxialMovingBeam_chirp.m for the full random-IC estimate).

clear all;

%% Dynamical system setup (nonlinear damping)
n = 10;
[mass,damp,gyro,stiff,fnl,fext] = build_model(n,'nonlinear_damp');

DS = DynamicalSystem();
set(DS,'M',mass,'C',damp+gyro,'K',stiff,'fnl',fnl);
set(DS.Options,'Emax',6,'Nmax',10,'notation','multiindex');
set(DS.Options,'RayleighDamping',false,'BaseExcitation',true);
[V,D,W] = DS.linear_spectral_analysis();

[V, lambda, W] = eig(full(DS.A),full(DS.B));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) ); % mass normalized VMs
DS.W = (W*diag(1./(sqrt(mu)')));
DS.lambda = diag(lambda);

%% Chirp base-excitation forcing
tspan = linspace(0,10,1000);
c0 = 1;
w0 = imag(D(1));
sig = sin(2*pi*(c0/2*tspan.^2 + w0*tspan));
f_0 = fext;
Fext_T = f_0.*sig;

h = tspan(2);
Fext = Fext_T;
nc_order = 1;

set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
DS.ncc_order = nc_order;

figure
plot(tspan,sig,'-','LineWidth',3,'color',[0.5 0 0.5])
xlabel('$t$ [s]','Interpreter','latex');
ylabel('$h(\Omega(t))$','Interpreter','latex');
xlim([0 10])

%% Compute the aSSM anchor trajectory
S = SSM(DS);
set(S.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Piecewise_exact','Ftype','aperiodic')
order = 10;
ticstart = tic;
[A_0_P,timerPE] = S.compute_anchor(order);
time_PE = toc(ticstart);

epsilon = 1;
order = 10;
[A_0_outP] = output_anchor(A_0_P,order,epsilon);
order = 1;
[A_0_outPL] = output_anchor(A_0_P,order,epsilon);

F_ext1 = griddedInterpolant(S.System.tforce(:),S.System.Fext.','linear');
F_ext = @(xa) epsilon*(((xa >=S.System.tforce(1)) & (xa <=S.System.tforce(end))).*F_ext1(xa)).';

%% Full-order time integration, for comparison against the anchor
[F, lambda, V, G, DG, G_nl] = functionFromTensors(DS.M, DS.C, DS.K, fnl);

startt = tic;
IC = zeros(2*n,1);
q0 = A_0_outP(1:n,1);
qd0 = A_0_outP(n+1:end,1);
tmax = tspan(end)-tspan(1);
qdd0 = -DS.M\(F_ext(0)) - DS.K\q0;

TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);
residual = @(q,qd,qdd,t) residual_nonlinear_slow( q, qd, qdd, t, DS.M,DS.K,DS.C,G_nl, F_ext);
TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);
Fnltime = toc(startt);

%% Plot: full-order trajectory vs. aSSM anchor prediction (first modal coordinate)
figure
indexR = 1;
t_sim = TI_NL.Solution.time;
plot(tspan,A_0_outPL(indexR,:),'-','LineWidth',3,'color','blue')
hold on
plot(t_sim,TI_NL.Solution.q(indexR,:),'-','LineWidth',3,'color','black')
hold on
plot(tspan,A_0_outP(indexR,:),'--','LineWidth',3,'color',[1 0 0])
grid on

xlabel('$t$ [s]','Interpreter','latex');
ylabel('$u_1(t)$ [-]','Interpreter','latex');
xlim([0 10])

%% Reconstruct and animate the beam's spatial deflection w(x,t)
x = linspace(0,1,100);
wwP = 0;
wwL = 0;
wwF = 0;
for ij = 1:n
    wwP = wwP + (sin(ij*pi*x).').*A_0_outP(ij,:);
    wwL = wwL + (sin(ij*pi*x).').*A_0_outPL(ij,:);
    wwF = wwF + (sin(ij*pi*x).').*TI_NL.Solution.q(ij,:);
end

hFig = figure('DefaultAxesFontSize',18);
clear movieVector

for ind = 1:1000
    clf
    plot(x,wwL(:,ind),'-','LineWidth',3,'color','blue')
    hold on
    plot(x,wwF(:,ind),'-','LineWidth',3,'color','black')
    hold on
    plot(x,wwP(:,ind),'--','LineWidth',3,'color','red')
    grid on
    xlabel('$x$ [m]','Interpreter','latex');
    ylabel('$w(x,t)$ [m]','Interpreter','latex');
    title(strcat('$t = $',num2str(round(tspan(ind),2),'%4.2f' ),''),'Interpreter','latex');
    legend('O(1) GSS','Full model','O(10) GSS','Interpreter','latex')
    xlim([0 1])
    ylim([-13*10^-3,13*10^-3])
    set(gcf,'color','white')
    set(hFig, 'Units' , 'Inches' );
    pos = get(hFig, 'Position' );
    set(hFig, 'PaperPositionMode' , 'Auto' , 'PaperUnits' , 'Inches' , 'PaperSize' ,[pos(3), pos(4)])
    set(gcf,'Renderer','painters')
    box on

    movieVector(ind) = getframe(hFig);
end

myWriter = VideoWriter('axialmovingbeam_increased_chirp_response','MPEG-4');
myWriter.FrameRate = 35;
open(myWriter);
for indj = 1:1000
    writeVideo(myWriter,movieVector(indj));
end
close(myWriter);

%% Forcing-weakness ratio r_w — single-IC check (undamped free response)
startt = tic;
q0 = real(W(1:n,1));
qd0 = 0*A_0_outP(n+1:end,1000);
tmax = tspan(end)-tspan(1);
qdd0 = -DS.K\q0;

TI_NL = ImplicitNewmark('timestep',h,'alpha',0.005,'RelTol',1e-05,'linear',false);
residual = @(q,qd,qdd,t) residual_nonlinear_uf( q, qd, qdd, t, DS.M,DS.K,DS.C,G_nl);
TI_NL.Integrate(q0,qd0,qdd0,tmax,residual);

Inter_Forces = DS.C*TI_NL.Solution.qd + DS.K*TI_NL.Solution.q + G_nl([TI_NL.Solution.q;TI_NL.Solution.qd]);
tSP1 = tspan;
tSP2 = TI_NL.Solution.time;
F_net = sqrt(sum(Inter_Forces.^2));
F_e = epsilon*sqrt(sum(Fext_T.^2));
rw = trapz(tSP1,F_e)/(max(tSP1))./(trapz(tSP2,F_net)/(max(tSP2)));
