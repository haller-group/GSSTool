fig = figure 
indexR = n-1;
indexRd = n-1;
plot(TI_NL.Solution.time,Full_F,'-','LineWidth',3,'color','red')

% plot(TI_NL.Solution.time,Anchor_Sol_O5(indexR,:),'-','LineWidth',3,'color',[0 1 0 0.3])
% hold on
% plot(TI_NL.Solution.time,Anchor_Sol_O7(indexR,:),'-','LineWidth',3,'color',[0 0 1 0.3])
% hold on
xlabel('$t \,[$s$]$','Interpreter','latex');
ylabel('$G(t) \,[$N$]$','Interpreter','latex');
% legend('$O(25)$ anchor','$O(11)$ anchor','Full order model','Interpreter','latex')
title('Forcing nature','Interpreter','latex')
xlim([0 200])
% ylim([-0.16 0.16])