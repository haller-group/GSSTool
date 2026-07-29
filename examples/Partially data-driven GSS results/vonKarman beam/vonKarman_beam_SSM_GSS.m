%%
% clear all
load('SSM_details_forcing_realiation2.mat','R0','W0','S','Forcing_history','Xwq')
% load weights for large autoencoder+LSTM model
load("0.70_cantilevered_nElements_20_latdim_11_test2_1.mat")
x_ground_truth_g = x_ground_truth;
x_e_processed = x_processed;


% load results for small autoencoder+LSTM model
load('weights_GSSpaper/0.50_y_decoded_test2_cantilevered_orig_nElements_20_latdim_6.mat')

[R0] = coeffs_lex2revlex(R0,'TaylorCoeff');
%%
T0 = 200; %% PSD frequency domain resolution is ~ 1/T0
nPoints=2^13*1; %% control the accuracy of numerical differential equation
tspan = linspace(0,T0,nPoints+1);
tspan = tspan.';
% anchor prediction on rom 
Lin = double(R0(1).coeffs);
DS = DynamicalSystem();
set(DS,'A',Lin,'B',eye(2),'F',R0(2:end));
set(DS.Options,'Emax',5,'Nmax',20,'notation','multiindex')
DS.order = 1;

[V, lambda, W] = eig(full(inv(DS.B)*DS.A));
mu = diag(W' * DS.B * V);
DS.V = V * diag( 1./ sqrt(mu) ); % mass normalized VMs
DS.W = (W*diag(1./(sqrt(mu)')));%O.V.';
DS.lambda = diag(lambda); % is now a vector
%%
F_ext1 = griddedInterpolant(tspan(:),Forcing_history.','linear');

U = S.E.adjointBasis';
Fext = - U * [F_ext1(tspan).'; sparse(60,max(size(tspan)))];
h = tspan(2);
set(DS,'Fext',Fext,'tforce',tspan,'tend',tspan(end)-tspan(1),'dh',h);
nc_order =1;
DS.ncc_order = nc_order;

%%
% Compute reduced GSS Taylor coefficients

Gs = SSM(DS);
set(Gs.Options, 'reltol', 0.1,'notation','multiindex','Solve_Method','Piecewise_exact_first_order')
time_PE = 0;

order = 30;
ticstart = tic;
[A_0_P,timerPE] = Gs.compute_anchor(order);
time_PE =time_PE+ toc(ticstart);
%% 
epsilon = 1;
order =20;

[A_0_outP] = output_anchor(A_0_P,order,epsilon);

order =1;

[A_0_outPL] = output_anchor(A_0_P,order,epsilon);

%%
% reduced prediction
%%
figure
plot(timerPE,imag(A_0_outP(1,:)),'color','red')
hold on 
plot(timerPE,imag(A_0_outPL(1,:)),'color','blue')


%%
% Lift reduced GSS to the physical space


z1 = zeros(1,length(timerPE)); z2=z1;
indexR = 59;
                for i = 1:length(timerPE)
                    AS = expand_autonomous(W0,2*60, A_0_outP(:,i));
                    AS1 = expand_autonomous(W0,2*60, A_0_outPL(:,i));

                    z1(i) = AS(indexR);
                    z2(i) = AS1(indexR);
                end
figure

plot(tspan,z1,'--','LineWidth',3,'color','red');
hold on
plot(tspan,z2,'--','LineWidth',3,'color','green');
hold on
xlabel('$t$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
ylim([min(z2) max(z2)])

indexR = 1;
c = A_0_P;
for ind =1:30
    if ~isempty(A_0_P(ind).coeffs)
        c(ind).coeffs = A_0_P(ind).coeffs(indexR,:); 
    else
        c(ind).coeffs = zeros(1,8193); 
    end    
end
L = 10;
M = 10;
N = L+M;

dq = [];
for ind = L+1:L+M
    dq = [dq;(c(ind).coeffs).'];
end 

forder = N;
Coeff = [];
for indf = 1:forder 
    if ~isempty(A_0_P(indf).coeffs)
        Coeff=[Coeff;A_0_P(indf).coeffs(indexR,:)]; 
    else
        Coeff=[Coeff;zeros(1,8193)];
      
    end 
  
end 

order = M;
B_coeff = [];
vec = Coeff;
vecB = -dq;
Bi = [];
for ui = 1:order 
     Bi = [Bi;flip(vec(ui:order+ui-1,:)).'];
end    
bs = pinv(Bi)*vecB;
B_coeff = [B_coeff,bs];
 

A_coeff =[];
for indj = 1:order 
    if indj == 1
        as = Coeff(indj,:);
    else
        as = Coeff(indj,:);
        for uiy = 1:indj-1
            as = as + B_coeff(uiy).*Coeff(indj-uiy,:);
        end    
    end 
    A_coeff =[A_coeff;as];
end 

% Compute vector Pade approximation of GSS
dim = indexR;
order = M;
[Pade1,PadeD,PadeN,Coeff_mat] = compute_padet_vector(A_0_P,epsilon,A_coeff,B_coeff,order,dim);


indexR = 2;
c = A_0_P;

for ind =1:30
    if ~isempty(A_0_P(ind).coeffs)
        c(ind).coeffs = A_0_P(ind).coeffs(indexR,:); 
    else
        c(ind).coeffs = zeros(1,8193); 
    end    
end

N = L+M;

dq = [];
for ind = L+1:L+M
    dq = [dq;(c(ind).coeffs).'];
end 

forder = N;
Coeff = [];
for indf = 1:forder 
    if ~isempty(A_0_P(indf).coeffs)
        Coeff=[Coeff;A_0_P(indf).coeffs(indexR,:)]; 
    else
        Coeff=[Coeff;zeros(1,8193)];
      
    end 
  
end

order = M;
B_coeff = [];
vec = Coeff;
vecB = -dq;
Bi = [];
for ui = 1:order 
     Bi = [Bi;flip(vec(ui:order+ui-1,:)).'];
end    
bs = pinv(Bi)*vecB;
B_coeff = [B_coeff,bs];
 

A_coeff =[];
for indj = 1:order 
    if indj == 1
        as = Coeff(indj,:);
    else
        as = Coeff(indj,:);
        for uiy = 1:indj-1
            as = as + B_coeff(uiy).*Coeff(indj-uiy,:);
        end    
    end 
    A_coeff =[A_coeff;as];
end 

dim = indexR;
order = M;
[Pade2,PadeD,PadeN,Coeff_mat] = compute_padet_vector(A_0_P,epsilon,A_coeff,B_coeff,order,dim);

indexR = 59;
padez = zeros(1,length(timerPE));
 for i = 1:length(timerPE)
                    AS = expand_autonomous(W0,2*60, [Pade1(i);Pade2(i)]);
                    padez(i) = AS(indexR);
 end

 %%
 % Plot predictions on unseen forcing
figure
tspan = linspace(0,T0,nPoints+1);
plot(tspan(:),x_ground_truth(:,indexR)/max(abs(x_ground_truth(:,indexR))),'-','LineWidth',3,'color',[0 0 0]);
hold on 
plot(tspan,padez/max(abs(padez)),'--','LineWidth',3,'color',[1    0    0]);
hold on
plot(tspan,x_processed(:,indexR)/max(abs(x_processed(:,indexR))),'-','LineWidth',3,'color',[0    1    0 0.5]);
hold on 
plot(tspan,x_e_processed(:,indexR)/max(abs(x_e_processed(:,indexR))),'-.','LineWidth',3,'color',[0, 0.5, 0]);

hold on
xlabel('$t$ [s]','Interpreter','latex')
ylabel('$\frac{x}{\|x(t)\|}$ [m], transverse (free end)','Interpreter','latex')
xlim([120 160])
title('Testing under unseen forcing realization')
legend('Full model (time integration)','Pade GSS','AE+LSTM (6D)','AE + LSTM (11D)','Interpreter','latex')

%%
% Simple NMTE Calculation for Normalized Trajectories

%% Normalize trajectories
lstm_start_idx = 1;
gt_normalized = x_ground_truth(:,indexR) / max(abs(x_ground_truth(:,indexR)));

padez_normalized = padez / max(abs(padez));
padez_normalized = padez_normalized.';
x_processed_normalized = x_processed(:,indexR) / max(abs(x_processed(:,indexR)));

%% Calculate NMTE for Padez
% NMTE = mean of absolute differences
nmte_padez = mean(abs(padez_normalized(lstm_start_idx:end) - gt_normalized(1:end)));

%% Calculate NMTE for AE+LSTM (x_processed)
% Only compare the overlapping time range

gt_lstm_range = gt_normalized;
nmte_lstm = mean(abs(x_processed_normalized - gt_lstm_range));
rte_lstm_ours = (abs(x_processed_normalized - gt_lstm_range));
load("Relative_error_Eleni_model.mat",'rte_lstm_eleni');
%%
figure
plot(tspan,rte_lstm_ours,'-','LineWidth',1,'color',[0.33, 0.00, 0.50 0.5]);
hold on 
plot(tspan,rte_lstm_eleni,'-','LineWidth',1,'color',[1.000,0.549,0.000 0.5]);
hold on
legend('AE + LSTM (Ours)','AE + LSTM (Eleni)','Interpreter','latex')
xlabel('$t$ [s]','Interpreter','latex')
ylabel('Relative error','Interpreter','latex')

%% Display Results
fprintf('\n===== Normalized Mean Trajectory Error (NMTE) =====\n\n');
fprintf('PADEZ:           NMTE = %.6f (%.4f%%)\n', nmte_padez, nmte_padez*100);
fprintf('AE+LSTM:         NMTE = %.6f (%.4f%%)\n', nmte_lstm, nmte_lstm*100);
fprintf('\n====================================================\n');

%% Plot with error annotations
figure('Position', [100, 100, 1000, 600]);
plot(tspan, gt_normalized, '-', 'LineWidth', 3, 'color', [0 0 0], 'DisplayName', 'Ground Truth');
hold on;
plot(tspan, padez_normalized, '--', 'LineWidth', 3, 'color', [1 0 0], ...
     'DisplayName', sprintf('Padez (NMTE: %.4f)', nmte_padez));
plot(tspan(lstm_start_idx:end), x_processed_normalized, '-', 'LineWidth', 3, ...
     'color', [0 1 0 0.4], 'DisplayName', sprintf('AE+LSTM (NMTE: %.4f)', nmte_lstm));
hold off;
xlabel('Time', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized State', 'FontSize', 12, 'FontWeight', 'bold');
title('Normalized Trajectories with NMTE', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11);


%%
% hero video

hFig = figure('DefaultAxesFontSize',18,'Position',[100, 100, 700, 300]);                       % Bring up new figure
delta = 0.5;

clear movieVector
ind =1;
indexR = 59;
Truth = Xwq(indexR,:)/max(abs(Xwq(indexR,:)));
GSS = padez/max(abs(padez));
LSTM_x = x_processed;

for j =1:(3893-2459+1)
i = j+0;

     clf
     
 
t_sim = tspan;
LSTM_t = tspan;

plot(t_sim(2459+j-1),Truth(2459+j-1),'.','MarkerSize',25,'color','black')
hold on
plot(t_sim(2459+j-1),GSS(2459+j-1),'.','MarkerSize',25,'color','red')
hold on
plot(LSTM_t(2459+j-1),LSTM_x(2459+j-1,indexR),'.','MarkerSize',25,'color','green')
hold on
plot(LSTM_t(2459:2459+j-1),LSTM_x(2459:2459+j-1,indexR),'-','LineWidth',3,'color',[0 1 0 0.5])
hold on
plot(t_sim(2459:2459+j-1),Truth(2459:2459+j-1),'-','LineWidth',3,'color','black')
hold on
plot(t_sim(2459:2459+j-1),GSS(2459:2459+j-1),'--','LineWidth',3,'color',[1 0 0])
hold on
string = strcat('$\frac{x(t)}{\|x(t)\|}$ transverse (free end)');
l = legend('Full model (time integration)','$\mathrm{Pade}[10,10]$ GSS $x^*(t)$','Autoencoder + LSTM prediction','Interpreter','latex');
l.Position = [0.1456 0.1970 0.4433 0.2111];
xlabel('$t$ [s]','Interpreter','latex');
ylabel(string,'Interpreter','latex');
xlim([80 95])
ylim([-1 1])

grid on
     

 
set(gcf,'color','white')

set(gcf,'color','white')
set(hFig, 'Units' , 'Inches' );
pos = get(hFig, 'Position' );
set(hFig, 'PaperPositionMode' , 'Auto' , 'PaperUnits' , 'Inches' , 'PaperSize' ,[pos(3), pos(4)])
set(gcf,'Renderer','painters')

    figssm = gcf;

 movieVector(ind) = getframe(hFig);
    
    ind = ind +1;

end

%%
myWriter = VideoWriter('x1_test_beam','MPEG-4');
myWriter.FrameRate = 50;

open(myWriter);

for indj = 1:ind-1
writeVideo(myWriter,movieVector(indj));
end

close(myWriter);
%%
load('SSM_beam_imp.mat','SSM_true','SSM_linear','SSM_taylor','SSM_pade')

figure

plot3(SSM_true(1,:),SSM_true(2,:),-SSM_true(3,:),'-','LineWidth',3,'color',[0 0 0])
hold on
plot3(SSM_pade(1,:),SSM_pade(2,:),-SSM_pade(3,:),'-','LineWidth',3,'color',[1    0    0 0.4])
ylim([min(SSM_linear(2,:)) max(SSM_linear(2,:))])
xlim([min(SSM_linear(1,:)) max(SSM_linear(1,:))])
legend('Full model (time integration)','$\mathrm{Pade}[10,10]$ GSS $x^*(t)$','Interpreter','latex')
box on 
grid on
xlabel('$x$','Interpreter','latex')
ylabel('$\dot{x}$','Interpreter','latex')
zlabel('$y$','Interpreter','latex')

%%
hFig = figure 
clear movieVector

 ind = 1;
tl = 30;
nssm = 30;
   [RHO,Theta]=meshgrid(linspace(0,0.003,nssm),linspace(0,2*pi,nssm));
     xq = linspace(-0.0005,0.0005,nssm);
     [Xq,Yq]=meshgrid(xq);
     ETA = RHO.*exp(1i*Theta);
     ETAq = Xq+ 1i*Yq;
     eta = ETAq(:);
     zSSMx = [];
     zSSMy = [];
     zSSMz = [];
     for indr = 1:length(eta)
                    AS = expand_autonomous(W0,60*2, [eta(indr);conj(eta(indr))]);

                    zSSMx = [zSSMx,AS(59)];
                    zSSMy = [zSSMy,AS(59+60)];
                    zSSMz = [zSSMz,AS(58)]; 
     end
st =2459-tl-1;
for j = 1:1
clf
i = j;
    
     Z1 = reshape(zSSMx,nssm,nssm);
     Z2 = reshape(zSSMy,nssm,nssm);
     Z3 = reshape(-zSSMz,nssm,nssm);

     h = surf(Z1,Z2,Z3);
     
       h.EdgeColor = 'black';
       h.EdgeAlpha = 0.5;
        h.FaceColor = 'cyan';
       h.FaceAlpha = 0.3;
      h.FaceLighting = 'gouraud';
      axis tight ; % Keep axis limits stable
       axis vis3d ; % Fix the aspect ratio for 3D rotation

     hold on 
     plot3(SSM_true(1,st+j:st+j+tl),SSM_true(2,st+j:st+j+tl),-SSM_true(3,st+j:st+j+tl),'-','LineWidth',3,'color',[0 0 0])
     hold on
     plot3(SSM_pade(1,st+j:st+j+tl),SSM_pade(2,st+j:st+j+tl),-SSM_pade(3,st+j:st+j+tl),'-','LineWidth',3,'color',[1    0    0 0.5])
     hold on 
     plot3(SSM_true(1,st+j+tl),SSM_true(2,st+j+tl),-SSM_true(3,st+j+tl),'.','MarkerSize',25,'color',[0 0 0])
     hold on
     plot3(SSM_pade(1,st+j+tl),SSM_pade(2,st+j+tl),-SSM_pade(3,st+j+tl),'.','MarkerSize',25,'color',[1    0    0])


    
grid on 

l = legend('SSM','Full model (time integration)','$\mathrm{Pade}[10,10]$ GSS $x^*(t)$','Interpreter','latex');
l.Position = [0.3858 0.8207 0.4193 0.1662];

xl = xlabel('$x$ [mm]','Interpreter','latex');
yl = ylabel('$\dot{x}$ [mm/s]','Interpreter','latex');
zl = zlabel('$z$ [nm]','Interpreter','latex');

yticks([-5*10^(-3)  0 5*10^(-3)])
yticklabels({ '-5' , '0', '5' })

xticks([-1*10^(-3)  0 1*10^(-3)])
xticklabels({ '-1','0', '1'})
zticks([0 1*10^(-7) 2*10^(-7) 3*10^(-7) 4*10^(-7) 5*10^(-7) 6*10^(-7) 7*10^(-7) 8*10^(-7)])
zticklabels({ '$0$','','','','','' , '' , '' ,'0.08'})


ztickangle(0)
ytickangle(0)
xtickangle(0)

fontsize(hFig,18, "points" )


   view(ind+300,17)
set(gcf,'color','white')
set(hFig, 'Units' , 'Inches' );
pos = get(hFig, 'Position' );
set(hFig, 'PaperPositionMode' , 'Auto' , 'PaperUnits' , 'Inches' , 'PaperSize' ,[pos(3), pos(4)])
set(gcf,'Renderer','painters')
set(gca, "TickLabelInterpreter" , 'latex' )
movieVector(ind) = getframe(hFig);

ind = ind +1;
end  

%%
myWriter = VideoWriter('beam_SSM_pade','MPEG-4');
myWriter.FrameRate = 50;

open(myWriter);

for indj = 1:length(movieVector)
writeVideo(myWriter,movieVector(indj));
end

close(myWriter);


%%
% final beam video 

GSS  = zeros(2*60,length(timerPE));
 for i = 1:length(timerPE)
                    AS = expand_autonomous(W0,2*60, [Pade1(i);Pade2(i)]);
                    GSS(:,i) = AS;
 end
GSS_pos = GSS(1:60,:)./(max(abs( GSS(1:60,:)).').');
Truth = Xwq(1:60,:)./(max(abs(Xwq(1:60,:)).').');
LSTM_x = (x_processed.').*(max(abs(Xwq(1:60,:)).').');

%%
GSS_pos = GSS(1:60,:);
Truth = Xwq(1:60,:);


%%

hFig = figure 
clear movieVector

 ind = 1;

numPoints = 60;
for j = 1000:1000
clf
i = j;
l = 1; % Beam length (along z)
b = 0.2; % Beam breadth (x-direction)
h = 0.1; % Beam height (y-direction)

u =   500*Truth(1:3:end,2459+j-1); % Linear axial displacement (along z)
w =   1000*Truth(2:3:end,2459+j-1); % Sinusoidal transverse displacement (in x)
theta =   1000*Truth(3:3:end,2459+j-1); % Sinusoidal rotation (about y-axis)
coordinates_true = [[0;u], [w(1)*100;w+w(1)*100], [theta(1);theta]];

u =  500*GSS_pos(1:3:end,2459+j-1); % Linear axial displacement (along z)
w =  1200*GSS_pos(2:3:end,2459+j-1); % Sinusoidal transverse displacement (in x)
theta = 1200*GSS_pos(3:3:end,2459+j-1); % Sinusoidal rotation (about y-axis)
coordinates_pred = [[0;u], [coordinates_true(1,2);w+coordinates_true(1,2)], [theta(1);theta]];


    h1=plotUprightVonKarmanBeam(l, b, h, coordinates_true, 21,[0.7 0.7 0.7],1);
    hold on
x = linspace(-0.5, 0.5, 50); % Adjust range and resolution as needed
y = linspace(-0.5, 0.5, 50);
[X, Y] = meshgrid(x, y); % Create a grid of x and y values

% Define z = 0 for all points
Z = zeros(size(X));

% Create the surface plot

h4 =surf(X+coordinates_true(1,2), Y, Z, 'FaceColor', [0.6 0.3 0], 'EdgeColor', 'none'); % Brown color

%     lj = legend([h1, h2, h3, h4], {'Full model', '$\mathrm{Pade}[10,10]$ GSS $x^*(t)$', 'AE+LSTM', 'Ground'}, 'Location', 'best','Interpreter','latex');

% l = legend('SSM','Full model (time integration)','$\mathrm{Pade}[10,10]$ GSS $x^*(t)$','Interpreter','latex');

xl = xlabel('$x$ (transverse)','Interpreter','latex');
yl = ylabel('$y$','Interpreter','latex');
zl = zlabel('$z$ (axial)','Interpreter','latex');


ztickangle(0)
ytickangle(0)
xtickangle(0)

fontsize(hFig,18, "points" )
axis([-0.7 0.7 -0.7 0.7 0 1])


set(gcf,'color','white')
set(hFig, 'Units' , 'Inches' );
pos = get(hFig, 'Position' );
set(hFig, 'PaperPositionMode' , 'Auto' , 'PaperUnits' , 'Inches' , 'PaperSize' ,[pos(3), pos(4)])
set(gcf,'Renderer','painters')
set(gca, "TickLabelInterpreter" , 'latex' )
movieVector(ind) = getframe(hFig);

ind = ind +1;
end  


%%
myWriter = VideoWriter('beam_SSMpade_video','MPEG-4');
myWriter.FrameRate = 50;

open(myWriter);

for indj = 1:length(movieVector)
writeVideo(myWriter,movieVector(indj));
end

close(myWriter);